import React, {useEffect, useState} from 'react';

type HistoryData = {
    [key: string]: string[];
};

type ResultsData = {
    [key: string]: {
        [key: string]: {
            [key: string]: number;
        };
    };
};

const HistoryPage = () => {
    const [historyData, setHistoryData] = useState<HistoryData>({});
    const [selectedFile, setSelectedFile] = useState<string>('');
    const [resultsData, setResultsData] = useState<ResultsData>({});
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
        const n = parseInt(str);
        return isNaN(n) ? 0 : n;
    };

    return (
        <div>
            <div className="border border-gray-200 rounded-md p-4 max-w-xs absolute top-8 right-8">
                <p className="text-lg font-semibold">
                    <a href="/" className="text-gray-800 no-underline hover:underline">Main Page</a>{" "}
                    <span role="img" aria-label="house">🏠</span>
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
                    <b>Select from the dropdown the appropriate attribute:</b>
                    <select value={selectedKey} onChange={(e) => setSelectedKey(e.target.value)}>
                        {headers.map((outerKey) => (
                            <option key={outerKey} value={outerKey}>
                                {outerKey}
                            </option>
                        ))}
                    </select>
                    {Object.keys(resultsData).length > 0 && (
                        <table>
                            <thead>
                            <tr>
                                <th>System's Decision Making Confidence (%)</th>
                                <th>Occurrences</th>
                            </tr>
                            </thead>
                            <tbody>
                            {Object.keys(resultsData[selectedKey])
                                .sort((a, b) => toNumber(a) - toNumber(b))
                                .map((innerKey, index, array) => {
                                    const current = resultsData[selectedKey][innerKey];
                                    const previous = index > 0 ? resultsData[selectedKey][array[index - 1]] : {};

                                    const occurrences: { [key: string]: number } = {};
                                    for (const [key, value] of Object.entries(current)) {
                                        if (!(key in previous)) {
                                            occurrences[key] = value;
                                        }
                                    }

                                    if (Object.keys(occurrences).length === 0) {
                                        return null;
                                    }

                                    const threshold = 100 - toNumber(innerKey);

                                    return (
                                        <tr key={innerKey}>
                                            <td colSpan={1}
                                                style={{borderTop: '1px solid black', borderRight: '1px solid black'}}>
                                                {threshold}
                                            </td>
                                            <td colSpan={2} style={{borderTop: '1px solid black'}}>
                                                <pre>{JSON.stringify(occurrences, null, 4)}</pre>
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
                </div>
            )}</div>);
};
export default HistoryPage;