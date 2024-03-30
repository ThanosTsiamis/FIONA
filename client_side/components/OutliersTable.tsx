import React from 'react';

type Data = {
    [key: string]: {
        [key: string]: {
            [key: string]: {
                [key: string]: number;
            };
        };
    };
};

const OutliersTable: React.FC<{ resultsData: Data, selectedKey: string }> = ({ resultsData, selectedKey }) => {
    // Helper function to convert a string to a number
    const toNumber = (str: string): number => {
        const n = parseFloat(str);
        return isNaN(n) ? 0 : Number(n.toFixed(4));
    };

    return (
        <>
            <h2 style={{ fontSize: '60px', marginTop: '20px', marginBottom: '20px' }}>Outliers</h2>
            {Object.keys(resultsData).length > 0 && (
                <table>
                    <thead>
                        <tr>
                            <th>System&apos;s Decision Making Confidence (%)</th>
                            <th>Generalised Strings</th>
                            <th>Occurrences</th>
                        </tr>
                    </thead>
                    <tbody>
                        {Object.keys(resultsData[selectedKey]['outliers'])
                            .sort((a, b) => toNumber(a) - toNumber(b))
                            .map((innerKey, index, array) => {
                                const current = resultsData[selectedKey]['outliers'][innerKey];
                                const previous = index > 0 ? resultsData[selectedKey]['outliers'][array[index - 1]] : {};

                                const occurrences: { [key: string]: number } = {};
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
                                            {Object.values(occurrences).map((value, index) => (
                                                <div key={value}>
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
