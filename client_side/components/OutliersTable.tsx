import React from 'react';
import {OutliersByThreshold, Results, ValueCounts} from '../lib/api';

const OutliersTable: React.FC<{ resultsData: Results, selectedKey: string }> = ({ resultsData, selectedKey }) => {
    // Helper function to convert a string to a number
    const toNumber = (str: string): number => {
        const n = parseFloat(str);
        return isNaN(n) ? 0 : Number(n.toFixed(4));
    };

    // selectedKey is not guaranteed to be a key of resultsData (e.g. before the fetch
    // resolves), so resolve the column defensively instead of indexing straight in.
    const outliersByThreshold: OutliersByThreshold | undefined =
        resultsData[selectedKey]?.outliers;

    return (
        <>
            <h2 style={{ fontSize: '60px', marginTop: '20px', marginBottom: '20px' }}>Outliers</h2>
            {outliersByThreshold && (
                <table>
                    <thead>
                        <tr>
                            <th>System&apos;s Decision Making Confidence (%)</th>
                            <th>Generalised Strings</th>
                            <th>Occurrences</th>
                        </tr>
                    </thead>
                    <tbody>
                        {Object.keys(outliersByThreshold)
                            .sort((a, b) => toNumber(a) - toNumber(b))
                            .map((innerKey, index, array) => {
                                const current = outliersByThreshold[innerKey];
                                const previous = index > 0 ? outliersByThreshold[array[index - 1]] : {};

                                // A representative maps to {value: count}, not a bare count, so the
                                // occurrences column renders the whole map (as it always has).
                                const occurrences: { [key: string]: ValueCounts } = {};
                                for (const [key, value] of Object.entries(current)) {
                                    if (!(key in previous)) {
                                        occurrences[key] = value;
                                    }
                                }

                                if (Object.keys(occurrences).length === 0) {
                                    return null;
                                }

                                const threshold = 100 - toNumber(innerKey); // Convert threshold to 100-threshold

                                return (
                                    <tr key={innerKey}>
                                        <td colSpan={1}
                                            style={{ borderTop: '1px solid black', borderRight: '1px solid black' }}>
                                            {threshold}
                                        </td>
                                        <td colSpan={1}
                                            style={{ borderTop: '1px solid black', borderRight: '1px solid black' }}>
                                            {Object.keys(occurrences).map(key => (
                                                <div key={key}>{key}</div>
                                            ))}
                                        </td>
                                        <td colSpan={1} className="border-t border-r border-black border-b">
                                            {Object.entries(occurrences).map(([key, value], index) => (
                                                <div key={key}>
                                                    {JSON.stringify(value)}
                                                    {index !== Object.values(occurrences).length - 1 && (
                                                        <hr className="border-dotted border-black my-1" />
                                                    )}
                                                </div>
                                            ))}
                                        </td>
                                    </tr>
                                );
                            })}
                    </tbody>
                    <tfoot>
                        <tr>
                            <td colSpan={2} style={{ borderTop: '1px solid black' }} />
                        </tr>
                    </tfoot>
                </table>
            )}
        </>
    );
};

export default OutliersTable;
