// Single source of truth for the backend base URL.
//
// Next.js only inlines variables prefixed with NEXT_PUBLIC_ into the client bundle;
// the REACT_APP_ prefix belongs to Create React App and is silently ignored here,
// which is why the URL used to be hardcoded in six places.
const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000';

export default API_URL;

/**
 * Shape produced by api/algorithm.py add_outlying_elements_to_attribute().
 *
 *   { "<column>": {
 *       "outliers": { "<threshold>": { "<representative>": { "<value>": <count> } } },
 *       "patterns": { "<threshold>": { "<generic>": { "<representative>": { "<value>": <count> } } } }
 *   } }
 */
export type ValueCounts = { [value: string]: number };
export type Representatives = { [representative: string]: ValueCounts };
export type OutliersByThreshold = { [threshold: string]: Representatives };
export type PatternsByThreshold = { [threshold: string]: { [generic: string]: Representatives } };

export interface ColumnResult {
    outliers: OutliersByThreshold;
    /** BriefSection only needs the outliers, so this is optional. */
    patterns?: PatternsByThreshold;
}

export type Results = { [column: string]: ColumnResult };

export const SUPPORTED_UPLOAD_EXTENSIONS = ['.csv', '.xlsx', '.json', '.tsv'] as const;
export const PREVIEWABLE_EXTENSIONS = ['.csv', '.tsv'] as const;

export function fileExtension(filename: string): string {
    const dot = filename.lastIndexOf('.');
    if (dot < 0) {
        return '';
    }
    return filename.slice(dot).toLowerCase();
}

export function isSupportedUpload(filename: string): boolean {
    return (SUPPORTED_UPLOAD_EXTENSIONS as readonly string[]).includes(fileExtension(filename));
}

export function isPreviewableDataset(filename: string): boolean {
    return (PREVIEWABLE_EXTENSIONS as readonly string[]).includes(fileExtension(filename));
}

/** Prefer the JSON `error` field the API returns; fall back to a status line. */
export async function readApiError(response: Response, fallback: string): Promise<string> {
    try {
        const payload = await response.json() as { error?: string };
        if (payload && typeof payload.error === 'string' && payload.error.trim()) {
            return payload.error;
        }
    } catch {
        // Body was not JSON; use the fallback below.
    }
    return `${fallback} (${response.status})`;
}
