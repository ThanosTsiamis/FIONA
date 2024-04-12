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
    const [expanded, setExpanded] = useState<string | null>(null);

    const toggleExpand = (patternKey: string) => {
        if (expanded === patternKey) {
            setExpanded(null);
        } else {
            setExpanded(patternKey);
        }
    };

    const isTooLong = (data: number) => Object.keys(data).length > 5; // Example threshold

    return (
        <>
            <h2 style={{fontSize: '60px', marginTop: '20px', marginBottom: '20px'}}>Patterns</h2>
            {resultsData[selectedKey] && resultsData[selectedKey]['patterns'] && (
                <table>
                    <thead>
                    <tr>
                        <th>Generic Patterns</th>
                        <th>Minimum Ensured Coverage</th>
                        <th>Specific Patterns</th>
                        <th>Minimum Ensured Coverage</th>
                    </tr>
                    </thead>
                    <tbody>
                    {Object.keys(resultsData[selectedKey]['patterns']).map((innerKey) => {
                        const patterns = resultsData[selectedKey]['patterns'][innerKey];

                        return Object.entries(patterns).map(([pattern, value]) => {
                            // Properly use reduce with type annotations for TypeScript
                            const sum = Object.values(value).reduce<number>((acc, nested) => {
                                return acc + Object.values(nested).reduce<number>((subAcc, num) => subAcc + Number(num), 0);
                            }, 0);

                            const specificPatterns = Object.keys(value);

                            return (
                                <tr key={pattern} onClick={() => toggleExpand(pattern)}>
                                    <td style={{border: '1px solid black', textAlign: 'center'}}>{pattern}</td>
                                    <td style={{border: '1px solid black', textAlign: 'center'}}>{sum.toFixed(5)}</td>
                                    <td style={{border: '1px solid black', textAlign: 'center'}}>
                                        {expanded === pattern || !isTooLong(value)
                                            ? specificPatterns.map(specificPattern => (
                                                <div key={specificPattern}>{specificPattern}</div>
                                            ))
                                            : <button onClick={() => toggleExpand(pattern)}>Click to expand</button>
                                        }
                                    </td>
                                    <td style={{border: '1px solid black', textAlign: 'center'}}>
                                        {expanded === pattern
                                            ? Object.entries(value).map(([specificPattern, specificPatternValue]) => (
                                                <div key={specificPattern}>
                                                    {Object.entries(specificPatternValue).map(([nestedPattern, nestedPatternValue]) => (
                                                        <div key={nestedPattern}>
                                                            {Number(nestedPatternValue).toFixed(5)}
                                                        </div>
                                                    ))}
                                                </div>
                                            ))
                                            : 'Click to expand'
                                        }
                                    </td>
                                </tr>
                            );
                        });
                    })}
                    </tbody>
                </table>
            )}
        </>
    );
};

export default PatternsTable;
