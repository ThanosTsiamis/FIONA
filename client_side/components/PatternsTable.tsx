import React, {useState} from 'react';

type Data = {
    [key: string]: {
        [key: string]: {
            [key: string]: {
                [key: string]: number;
            };
        };
    };
};

const PatternsTable: React.FC<{ resultsData: Data, selectedKey: string }> = ({resultsData, selectedKey}) => {
    const [expandedRow, setExpandedRow] = useState<string | null>(null);

    const toggleRow = (pattern: string) => {
        if (expandedRow === pattern) {
            setExpandedRow(null);
        } else {
            setExpandedRow(pattern);
        }
    };

    return (
        <>
            <h2 style={{fontSize: '60px', marginTop: '20px', marginBottom: '20px'}}>Patterns</h2>
            {resultsData[selectedKey] && resultsData[selectedKey]['patterns'] && (
                <table>
                    <thead>
                    <tr>
                        <th>Generic Patterns</th>
                        <th>Minimum Ensured Coverage</th>
                    </tr>
                    </thead>
                    <tbody>
                    {Object.keys(resultsData[selectedKey]['patterns']).map((innerKey, index, array) => {
                        const current = resultsData[selectedKey]['patterns'][innerKey];
                        const previous = index < array.length - 1 ? resultsData[selectedKey]['patterns'][array[index + 1]] : {};
                        const patterns: { [key: string]: number } = {};

                        for (const [key, value] of Object.entries(current)) {
                            if (!(key in previous)) {
                                let sum = 0;
                                for (const nestedValue of Object.values(value)) {
                                    for (const numericValue of Object.values(nestedValue)) {
                                        sum += Number(numericValue);
                                    }
                                }
                                patterns[key] = sum;
                            }
                        }

                        if (Object.keys(patterns).length === 0) {
                            return null;
                        }

                        return (
                            <>
                                {Object.entries(patterns).map(([pattern, value]) => {
                                    const sum = value;
                                    const isExpanded = expandedRow === pattern;
                                    return (
                                        <React.Fragment key={pattern}>
                                            <tr onClick={() => toggleRow(pattern)}>
                                                <td style={{
                                                    border: '1px solid black',
                                                    textAlign: 'center'
                                                }}>{pattern}</td>
                                                <td style={{
                                                    border: '1px solid black',
                                                    textAlign: 'center'
                                                }}>{sum.toFixed(5)}</td>
                                            </tr>
                                            {isExpanded && (
                                                <tr>
                                                    <td colSpan={2}>
                                                        {Object.entries(current[pattern]).map(([specificPattern, specificPatternValue]) => (
                                                            <div key={specificPattern}>
                                                                <strong>{specificPattern}</strong>
                                                                {Object.entries(specificPatternValue).map(([nestedPattern, nestedPatternValue]) => (
                                                                    <div key={nestedPattern}>
                                                                        {nestedPattern}: {Number(nestedPatternValue).toFixed(5)}
                                                                    </div>
                                                                ))}
                                                            </div>
                                                        ))}
                                                    </td>
                                                </tr>
                                            )}
                                        </React.Fragment>
                                    );
                                })}
                            </>
                        );
                    })}
                    </tbody>
                </table>
            )}
        </>
    );
};

export default PatternsTable;
