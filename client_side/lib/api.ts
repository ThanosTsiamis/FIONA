const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:5000';

export class ApiError extends Error {
    status: number;

    constructor(message: string, status: number) {
        super(message);
        this.name = 'ApiError';
        this.status = status;
    }
}

async function parseJsonResponse<T>(response: Response): Promise<T> {
    const contentType = response.headers.get('content-type') || '';
    const payload = contentType.includes('application/json') ? await response.json() : null;

    if (!response.ok) {
        const message =
            payload && typeof payload === 'object' && 'error' in payload && typeof payload.error === 'string'
                ? payload.error
                : `Request failed with status ${response.status}.`;
        throw new ApiError(message, response.status);
    }

    return payload as T;
}

export async function fetchJson<T>(path: string): Promise<T> {
    const response = await fetch(`${API_BASE_URL}${path}`);
    return parseJsonResponse<T>(response);
}

export async function uploadDataset(formData: FormData): Promise<Response> {
    const response = await fetch(`${API_BASE_URL}/api/upload`, {
        method: 'POST',
        body: formData,
        redirect: 'follow',
    });

    if (!response.ok && !response.redirected) {
        await parseJsonResponse(response);
    }

    return response;
}
