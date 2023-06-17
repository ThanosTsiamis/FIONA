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
    const [isLoadingFiles, setIsLoadingFiles] = useState<boolean>(false);
    const [isLoadingAttributes, setIsLoadingAttributes] = useState<boolean>(false);
    const [loadingProgress, setLoadingProgress] = useState<number>(0);

    useEffect(() => {
        const fetchHistoryData = async () => {
            setIsLoadingFiles(true);
            const response = await fetch('http://localhost:5000/api/history');
            const jsonData = await response.json();
            setHistoryData(jsonData);
            setIsLoadingFiles(false);
        };

        fetchHistoryData();
    }, []);

    useEffect(() => {
        const fetchResultsData = async () => {
            setIsLoadingAttributes(true);
            setLoadingProgress(0);
            const response = await fetch(`http://localhost:5000/api/fetch/${selectedFile}`);
            const totalBytes = response.headers.get('content-length');
            const reader = response.body!.getReader();
            let receivedBytes = 0;
            let chunks: Uint8Array[] = [];

            while (true) {
                const {done, value} = await reader.read();
                if (done) break;

                chunks.push(value);
                receivedBytes += value.length;

                if (totalBytes) {
                    const progress = (receivedBytes / Number(totalBytes)) * 100;
                    setLoadingProgress(progress);
                }
            }

            const concatenatedChunks = new Uint8Array(receivedBytes);
            let offset = 0;
            for (const chunk of chunks) {
                concatenatedChunks.set(chunk, offset);
                offset += chunk.length;
            }

            const decoder = new TextDecoder();
            const jsonData = JSON.parse(decoder.decode(concatenatedChunks));
            setResultsData(jsonData);
            setHeaders(Object.keys(jsonData));
            setSelectedKey(Object.keys(jsonData)[0]); // Select first outer key by default
            setIsLoadingAttributes(false);
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
            <div>
                <b>Select the JSON file:</b>
                {isLoadingFiles ? (
                    <div>Loading files...</div> // Render a loading indicator for file loading
                ) : (
                    <select value={selectedFile} onChange={(e) => setSelectedFile(e.target.value)}>
                        <option value="">-- Select a file --</option>
                        {Object.keys(historyData).map((key) => (
                            <option key={key} value={historyData[key]}>
                                {historyData[key]}
                            </option>
                        ))}
                    </select>
                )}
            </div>

            {selectedFile && (
                <div>
                    <b>Select from the dropdown the Appropriate Attribute:</b>
                    {isLoadingAttributes ? (
                        <div>Loading attributes...</div> // Render a loading indicator for attribute loading
                    ) : (
                        <select value={selectedKey} onChange={(e) => setSelectedKey(e.target.value)}>
                            {headers.map((outerKey) => (
                                <option key={outerKey} value={outerKey}>
                                    {outerKey}
                                </option>
                            ))}
                        </select>
                    )}
                    {isLoadingAttributes && (
                        <div className="relative">
                            <progress
                                value={loadingProgress}
                                max={100}
                                className="w-full h-2 bg-blue-200"
                            />
                            <div className="absolute inset-0 flex items-center justify-center text-white font-bold">
                                {loadingProgress.toFixed(0)}%
                            </div>
                        </div>
                    )}

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
                                <th>Minimum Coverage</th>
                            </tr>
                            </thead>
                            <tbody>
                            {Object.keys(resultsData[selectedKey]['patterns']).map((innerKey, index, array) => {
                                const current = resultsData[selectedKey]['patterns'][innerKey];
                                const previous =
                                    index < array.length - 1 ? resultsData[selectedKey]['patterns'][array[index + 1]] : {};

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
                                            return (
                                                <tr key={pattern}>
                                                    <td style={{
                                                        border: '1px solid black',
                                                        textAlign: 'center'
                                                    }}>{pattern}</td>
                                                    <td style={{border: '1px solid black', textAlign: 'center'}}>
                                                        {sum.toFixed(5)}
                                                    </td>
                                                    <td style={{border: '1px solid black', textAlign: 'center'}}>
                                                        {Object.keys(current[pattern]).map((specificPattern) => (
                                                            <div key={specificPattern}>{specificPattern}</div>
                                                        ))}
                                                    </td>
                                                    <td style={{border: '1px solid black', textAlign: 'center'}}>
                                                        {Object.entries(current[pattern]).map(([specificPattern, specificPatternValue]) => (
                                                            <div key={specificPattern}>
                                                                {Object.entries(specificPatternValue).map(([nestedPattern, nestedPatternValue]) => (
                                                                    <div key={nestedPattern}>
                                                                        {Number(nestedPatternValue).toFixed(5)}
                                                                    </div>
                                                                ))}
                                                            </div>
                                                        ))}
                                                    </td>
                                                </tr>
                                            );
                                        })}
                                    </>
                                );
                            })}
                            </tbody>
                        </table>
                    )}


                </div>
            )}
        </div>
    );
};

export default HistoryPage;
