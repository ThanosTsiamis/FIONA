import React, {useEffect, useState} from 'react';

type HistoryData = {
    [key: string]: string[];
};
type Data = {
    [key: string]: {
        [key: string]: {
            [key: string]: {
                [key: string]: number;
            };
        };
    };
};

const HistoryPage = () => {
    const [historyData, setHistoryData] = useState<HistoryData>({});
    const [selectedFile, setSelectedFile] = useState<string>('');
    const [resultsData, setResultsData] = useState<Data>({});
    const [headers, setHeaders] = useState<string[]>([]);
    const [selectedKey, setSelectedKey] = useState<string>('');

    useEffect(() => {
        const fetchHistoryData = async () => {
            const response = await fetch('http://localhost:5000/api/history');
            const jsonData = await response.json();
            setHistoryData(jsonData);
        };

        fetchHistoryData();
    }, []);

    useEffect(() => {
        const fetchResultsData = async () => {
            const response = await fetch(`http://localhost:5000/api/fetch/${selectedFile}`);
            const jsonData = await response.json();
            setResultsData(jsonData);
            setHeaders(Object.keys(jsonData));
            setSelectedKey(Object.keys(jsonData)[0]); // Select first outer key by default
        };

        if (selectedFile) {
            fetchResultsData();
        }
    }, [selectedFile]);

    // Helper function to convert a string to a number
    const toNumber = (str: string): number => {
        const n = parseFloat(str);
        return isNaN(n) ? 0 : Number(n.toFixed(4));
    };

    return (
        <div>
            <div className="border border-gray-200 rounded-md p-4 max-w-xs absolute top-8 right-8">
                <p className="text-lg font-semibold">
                    <a href="/" className="text-gray-800 no-underline hover:underline">
                        Main Page
                    </a>{' '}
                    <span role="img" aria-label="house">
            🏠
          </span>
                </p>
            </div>

            <b>Select the JSON file:</b>
            <select value={selectedFile} onChange={(e) => setSelectedFile(e.target.value)}>
                <option value="">-- Select a file --</option>
                {Object.keys(historyData).map((key) => (
                    <option key={key} value={historyData[key]}>
                        {historyData[key]}
                    </option>
                ))}
            </select>

            {selectedFile && (
                <div>
                    <b>Select from the dropdown the Appropriate Attribute</b>
                    <select value={selectedKey} onChange={(e) => setSelectedKey(e.target.value)}>
                        {headers.map((outerKey) => (
                            <option key={outerKey} value={outerKey}>
                                {outerKey}
                            </option>
                        ))}
                    </select>
                    <h2 style={{fontSize: '60px', marginTop: '20px', marginBottom: '20px'}}>Outliers</h2>
                    {Object.keys(resultsData).length > 0 && (
                        <table>
                            <thead>
                            <tr>
                                <th>System's Decision Making Confidence (%)</th>
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
                                                style={{borderTop: '1px solid black', borderRight: '1px solid black'}}>
                                                {threshold}
                                            </td>
                                            <td colSpan={1}
                                                style={{borderTop: '1px solid black', borderRight: '1px solid black'}}>
                                                {Object.keys(occurrences).map(key => (
                                                    <div key={key}>{key}</div>
                                                ))}
                                            </td>
                                            <td colSpan={1} className="border-t border-r border-black border-b">
                                                {Object.values(occurrences).map((value, index) => (
                                                    <div key={value}>
                                                        {JSON.stringify(value)}
                                                        {index !== Object.values(occurrences).length - 1 && (
                                                            <hr className="border-dotted border-black my-1"/>
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
                                <td colSpan={2} style={{borderTop: '1px solid black'}}/>
                            </tr>
                            </tfoot>
                        </table>
                    )}

                    <h2 style={{fontSize: '60px', marginTop: '20px', marginBottom: '20px'}}>Patterns</h2>
                    {resultsData[selectedKey] && resultsData[selectedKey]['patterns'] && (
                        <table>
                            <thead>
                            <tr>
                                <th>Generic Patterns</th>
                                <th>Minimum Coverage</th>
                                <th>Specific Patterns</th>
                                <th>Minimum coverage</th>
                            </tr>
                            </thead>
                            <tbody>
                            {Object.keys(resultsData[selectedKey]['patterns']).map((innerKey, index, array) => {
                                const current = resultsData[selectedKey]['patterns'][innerKey];
                                const previous =
                                    index < array.length - 1
                                        ? resultsData[selectedKey]['patterns'][array[index + 1]]
                                        : {};

                                const patterns: { [key: string]: number } = {};
                                for (const [key, value] of Object.entries(current)) {
                                    if (!(key in previous)) {
                                        patterns[key] = value;
                                    }
                                }

                                if (Object.keys(patterns).length === 0) {
                                    return null;
                                }

                                return (
                                    <>
                                        {Object.entries(patterns).map(([pattern, value]) => (
                                            <tr key={pattern}>
                                                <td style={{
                                                    border: '1px solid black',
                                                    textAlign: 'center'
                                                }}>{pattern}</td>
                                                <td style={{border: '1px solid black', textAlign: 'center'}}
                                                >Athroisma
                                                </td>
                                                <td style={{
                                                    border: '1px solid black',
                                                    textAlign: 'center'
                                                }}>
                                                    {Object.keys(value).map((specific_patterns) =>
                                                        <div>
                                                            {specific_patterns}
                                                        </div>
                                                    )}
                                                </td>
                                                <td style={{
                                                    border: '1px solid black',
                                                    textAlign: 'center'
                                                }}>

                                                    Mikra Athroismata
                                                </td>
                                            </tr>
                                        ))}
                                    </>
                                );
                            })}
                            </tbody>
                        </table>)}

                </div>
            )}
        </div>
    );
};

export default HistoryPage;
