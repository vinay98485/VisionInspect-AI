export const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

export class AuthError extends Error {
  constructor(message = "Your session has expired. Please log in again.") {
    super(message);
    this.name = "AuthError";
    this.status = 401;
  }
}

export class ForbiddenError extends Error {
  constructor(message = "You do not have permission to perform this action.") {
    super(message);
    this.name = "ForbiddenError";
    this.status = 403;
  }
}

export class NotFoundError extends Error {
  constructor(message = "The requested resource was not found.") {
    super(message);
    this.name = "NotFoundError";
    this.status = 404;
  }
}

export class ApiError extends Error {
  constructor(message, status = 400, details = null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.details = details;
  }
}

export function formatErrorMessage(errorData, fallback = "An unexpected error occurred.") {
  if (!errorData) return fallback;
  if (typeof errorData === "string") return errorData;

  if (errorData.detail) {
    if (typeof errorData.detail === "string") {
      return errorData.detail;
    }
    if (Array.isArray(errorData.detail)) {
      return errorData.detail
        .map((err) => {
          const field = Array.isArray(err.loc) ? err.loc.slice(-1)[0] : "";
          const msg = err.msg || "Invalid value";
          if (field && field !== "body") {
            const readableField = field.replace(/_/g, " ");
            return `${readableField}: ${msg}`;
          }
          return msg;
        })
        .join(". ");
    }
    if (typeof errorData.detail === "object") {
      return JSON.stringify(errorData.detail);
    }
  }

  if (errorData.message) return errorData.message;
  return fallback;
}

export async function apiFetch(endpoint, options = {}) {
  const url = endpoint.startsWith("http") ? endpoint : `${API_BASE}${endpoint}`;
  const token = localStorage.getItem("access_token");

  const headers = new Headers(options.headers || {});

  if (token && !headers.has("Authorization")) {
    headers.set("Authorization", `Bearer ${token}`);
  }

  let response;
  try {
    response = await fetch(url, {
      ...options,
      headers,
    });
  } catch {
    throw new ApiError(
      "Unable to connect to the server. Please ensure the backend is running.",
      0
    );
  }

  if (response.status === 401) {
    localStorage.removeItem("access_token");
    localStorage.removeItem("current_user");
    throw new AuthError("Your session has expired. Please log in again.");
  }

  if (response.status === 403) {
    let errorMsg = "You do not have permission to perform this action.";
    try {
      const data = await response.json();
      errorMsg = formatErrorMessage(data, errorMsg);
    } catch {
      // Keep default message
    }
    throw new ForbiddenError(errorMsg);
  }

  if (response.status === 404) {
    let errorMsg = "Resource not found.";
    try {
      const data = await response.json();
      errorMsg = formatErrorMessage(data, errorMsg);
    } catch {
      // Keep default message
    }
    throw new NotFoundError(errorMsg);
  }

  return response;
}

export async function login(email, password) {
  let response;
  try {
    response = await fetch(`${API_BASE}/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    });
  } catch {
    throw new ApiError(
      "Unable to connect to the server. Please ensure the backend is running.",
      0
    );
  }

  const data = await response.json();

  if (!response.ok) {
    throw new ApiError(
      formatErrorMessage(data, "Invalid email or password."),
      response.status
    );
  }

  const token = data.access_token || data["access token"];
  if (token) {
    localStorage.setItem("access_token", token);
  }
  if (data.user) {
    localStorage.setItem("current_user", JSON.stringify(data.user));
  }

  return data;
}

export async function register(userData) {
  let response;
  try {
    response = await fetch(`${API_BASE}/auth/register`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(userData),
    });
  } catch {
    throw new ApiError(
      "Unable to connect to the server. Please ensure the backend is running.",
      0
    );
  }

  const data = await response.json();

  if (!response.ok) {
    throw new ApiError(
      formatErrorMessage(data, "Registration failed."),
      response.status
    );
  }

  return data;
}

export async function getCurrentUser() {
  const response = await apiFetch("/auth/me");
  const data = await response.json();
  if (!response.ok) {
    throw new ApiError(formatErrorMessage(data, "Failed to load user profile."), response.status);
  }
  localStorage.setItem("current_user", JSON.stringify(data));
  return data;
}

export async function getImages() {
  const response = await apiFetch("/images/");
  const data = await response.json();
  if (!response.ok) {
    throw new ApiError(formatErrorMessage(data, "Failed to load images."), response.status);
  }
  return data;
}

export async function getSupervisorReviewQueue() {
  const response = await apiFetch("/images/supervisor/review-queue");
  const data = await response.json();
  if (!response.ok) {
    throw new ApiError(formatErrorMessage(data, "Failed to load review queue."), response.status);
  }
  return data;
}

export async function getImage(imageId) {
  const response = await apiFetch(`/images/${imageId}`);
  const data = await response.json();
  if (!response.ok) {
    throw new ApiError(formatErrorMessage(data, "Failed to load image details."), response.status);
  }
  return data;
}

export async function getImageBlobUrl(imageId) {
  const response = await apiFetch(`/images/${imageId}/file`);
  if (!response.ok) {
    throw new ApiError("Failed to fetch image file.", response.status);
  }
  const blob = await response.blob();
  return URL.createObjectURL(blob);
}
export async function getInspectionOverlayBlobUrl(imageId) {
  const response = await apiFetch(
    `/images/${imageId}/inspection-overlay`
  );

  if (!response.ok) {
    const data = await response.json().catch(() => ({}));

    throw new ApiError(
      formatErrorMessage(
        data,
        "Failed to load AI defect overlay."
      ),
      response.status
    );
  }

  const blob = await response.blob();

  return URL.createObjectURL(blob);
}

export async function uploadImage(file) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await apiFetch("/images/upload", {
    method: "POST",
    body: formData,
  });

  const data = await response.json();
  if (!response.ok) {
    throw new ApiError(formatErrorMessage(data, "Image upload failed."), response.status);
  }
  return data;
}
export async function inspectImage(imageId, category = null, defectType = null) {
  const payload = {};
  if (category && category.trim()) {
    payload.category = category.trim();
  }
  if (defectType && defectType.trim()) {
    payload.defect_type = defectType.trim();
  }

  const response = await apiFetch(`/images/${imageId}/inspect`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  const data = await response.json();

  if (!response.ok) {
    throw new ApiError(
      formatErrorMessage(data, "Image inspection failed."),
      response.status
    );
  }

  return data;
}
export async function reviewImage(imageId, decision, notes = "") {
  const response = await apiFetch(`/images/${imageId}/review`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ decision, notes }),
  });

  const data = await response.json();
  if (!response.ok) {
    throw new ApiError(formatErrorMessage(data, "Review submission failed."), response.status);
  }
  return data;
}

export async function getAnalyticsSummary() {
  const response = await apiFetch("/analytics/summary");
  const data = await response.json();

  if (!response.ok) {
    throw new ApiError(
      formatErrorMessage(data, "Failed to load analytics."),
      response.status
    );
  }

  return data;
}

export async function getAnalyticsTrends(days = "30") {
  const response = await apiFetch(
    `/analytics/trends?days=${encodeURIComponent(days)}`
  );
  const data = await response.json();

  if (!response.ok) {
    throw new ApiError(
      formatErrorMessage(data, "Failed to load analytics trends."),
      response.status
    );
  }

  return data;
}

export async function exportQualityReport(format = "csv") {
  const response = await apiFetch(
    `/analytics/reports/export?format=${encodeURIComponent(format)}`
  );

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new ApiError(
      formatErrorMessage(errorData, "Failed to export quality report."),
      response.status
    );
  }

  let filename = `production_quality_report_${Date.now()}.${format}`;
  const disposition = response.headers.get("Content-Disposition");
  if (disposition) {
    const match = disposition.match(/filename=["']?([^"';]+)["']?/i);
    if (match && match[1]) {
      filename = match[1].trim();
    }
  }

  const blob = await response.blob();
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.style.display = "none";
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => {
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
  }, 100);

  return { filename };
}

export function getStoredUser() {
  const raw = localStorage.getItem("current_user");
  if (!raw) return null;
  try {
    return JSON.parse(raw);
  } catch {
    localStorage.removeItem("current_user");
    return null;
  }
}

export function logout() {
  localStorage.removeItem("access_token");
  localStorage.removeItem("current_user");
}