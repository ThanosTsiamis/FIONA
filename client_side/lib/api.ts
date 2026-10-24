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
