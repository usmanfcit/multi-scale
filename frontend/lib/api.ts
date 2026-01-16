const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface SearchHit {
  pinecone_id: string;
  score: number;
  image_url?: string | null;
  product_url?: string | null;
  name_english?: string | null;
  name_arabic?: string | null;
  category?: string | null;
  price_amount?: number | null;
  price_unit?: string | null;
  is_active?: boolean | null;
  store_id?: number | null;
  countries?: string[] | null;
  store?: string | null;
}

export interface SearchResponse {
  query_category?: string | null;
  hits: SearchHit[];
  message?: string | null;
}

export interface CatalogUpsertResponse {
  sku_id: string;
  image_id: string;
  upserted: boolean;
}

export interface BBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface DetectedObject {
  category: string;
  bbox: BBox;
  score: number;
  object_id: number;
}

export interface DetectionResponse {
  objects: DetectedObject[];
  image_width: number;
  image_height: number;
}

export interface SegmentedObject {
  category: string;
  bbox: BBox;
  score: number;
  object_id: number;
  mask_base64: string; // Base64 encoded PNG mask
}

export interface DetectionSegmentationResponse {
  objects: SegmentedObject[];
  image_width: number;
  image_height: number;
}

export interface HealthResponse {
  status: string;
}

export class ApiError extends Error {
  constructor(
    message: string,
    public status?: number,
    public response?: Response
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let errorMessage = `HTTP error! status: ${response.status}`;

    try {
      const contentType = response.headers.get("content-type");
      if (contentType && contentType.includes("application/json")) {
        const errorData = await response.json();
        // FastAPI returns errors in format: { "detail": "error message" }
        // Handle both string and array formats
        if (typeof errorData.detail === "string") {
          errorMessage = errorData.detail;
        } else if (Array.isArray(errorData.detail)) {
          // Handle validation errors array
          const firstError = errorData.detail[0];
          if (firstError && typeof firstError === "object") {
            const field = firstError.loc?.join(".") || "field";
            errorMessage = `Validation error for ${field}: ${firstError.msg || "Invalid value"}`;
          } else {
            errorMessage = errorData.detail.join(", ");
          }
        } else if (errorData.message) {
          errorMessage = errorData.message;
        } else {
          errorMessage = JSON.stringify(errorData);
        }
      } else {
        const errorText = await response.text();
        if (errorText) {
          errorMessage = errorText;
        }
      }
    } catch {
      // If parsing fails, use default error message
    }

    throw new ApiError(errorMessage, response.status, response);
  }
  return response.json();
}

export async function healthCheck(): Promise<HealthResponse> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/v1/health`);
    return handleResponse<HealthResponse>(response);
  } catch (error) {
    if (error instanceof TypeError && error.message.includes("fetch")) {
      throw new ApiError(
        `Unable to connect to the server. Please check if the backend is running at ${API_BASE_URL}`,
        0
      );
    }
    if (error instanceof ApiError) {
      throw error;
    }
    throw new ApiError(
      error instanceof Error ? error.message : "An unexpected error occurred",
      0
    );
  }
}

export async function searchProducts(
  image: File,
  options?: {
    category?: string;
    top_k?: number;
  }
): Promise<SearchResponse> {
  const formData = new FormData();
  formData.append("image", image);

  if (options?.category) {
    formData.append("assigned_category", options.category);
  }

  // Always send top_k (backend defaults to 20 if not provided)
  const topK = options?.top_k && options.top_k > 0 ? options.top_k : 20;
  formData.append("top_k", topK.toString());

  try {
    const response = await fetch(`${API_BASE_URL}/api/v1/search`, {
      method: "POST",
      body: formData,
    });

    return handleResponse<SearchResponse>(response);
  } catch (error) {
    // Handle network errors (connection refused, CORS, etc.)
    if (error instanceof TypeError && error.message.includes("fetch")) {
      throw new ApiError(
        `Unable to connect to the server. Please check if the backend is running at ${API_BASE_URL}`,
        0
      );
    }
    // Re-throw if it's already an ApiError
    if (error instanceof ApiError) {
      throw error;
    }
    // Handle other errors
    throw new ApiError(
      error instanceof Error ? error.message : "An unexpected error occurred",
      0
    );
  }
}

export async function upsertCatalogItem(
  skuId: string,
  category: string,
  image: File,
  attributes?: Record<string, any>
): Promise<CatalogUpsertResponse> {
  const formData = new FormData();
  formData.append("sku_id", skuId);
  formData.append("category", category);
  formData.append("image", image);

  if (attributes) {
    formData.append("attributes_json", JSON.stringify(attributes));
  }

  try {
    const response = await fetch(`${API_BASE_URL}/api/v1/catalog/upsert`, {
      method: "POST",
      body: formData,
    });

    return handleResponse<CatalogUpsertResponse>(response);
  } catch (error) {
    // Handle network errors (connection refused, CORS, etc.)
    if (error instanceof TypeError && error.message.includes("fetch")) {
      throw new ApiError(
        `Unable to connect to the server. Please check if the backend is running at ${API_BASE_URL}`,
        0
      );
    }
    // Re-throw if it's already an ApiError
    if (error instanceof ApiError) {
      throw error;
    }
    // Handle other errors
    throw new ApiError(
      error instanceof Error ? error.message : "An unexpected error occurred",
      0
    );
  }
}

export async function detectObjects(
  image: File
): Promise<DetectionResponse> {
  const formData = new FormData();
  formData.append("image", image);

  try {
    const response = await fetch(`${API_BASE_URL}/api/v1/detect`, {
      method: "POST",
      body: formData,
    });

    return handleResponse<DetectionResponse>(response);
  } catch (error) {
    if (error instanceof TypeError && error.message.includes("fetch")) {
      throw new ApiError(
        `Unable to connect to the server. Please check if the backend is running at ${API_BASE_URL}`,
        0
      );
    }
    if (error instanceof ApiError) {
      throw error;
    }
    throw new ApiError(
      error instanceof Error ? error.message : "An unexpected error occurred",
      0
    );
  }
}

export async function detectAndSegmentObjects(
  image: File
): Promise<DetectionSegmentationResponse> {
  const formData = new FormData();
  formData.append("image", image);

  try {
    const response = await fetch(`${API_BASE_URL}/api/v1/detect-and-segment`, {
      method: "POST",
      body: formData,
    });

    return handleResponse<DetectionSegmentationResponse>(response);
  } catch (error) {
    if (error instanceof TypeError && error.message.includes("fetch")) {
      throw new ApiError(
        `Unable to connect to the server. Please check if the backend is running at ${API_BASE_URL}`,
        0
      );
    }
    if (error instanceof ApiError) {
      throw error;
    }
    throw new ApiError(
      error instanceof Error ? error.message : "An unexpected error occurred",
      0
    );
  }
}

