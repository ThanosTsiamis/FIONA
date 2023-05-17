import React, {useContext, useEffect, useState} from 'react';
import {UploadContext} from '../components/UploadContext';

type Data = {
    [key: string]: {
        [key: string]: {
            [key: string]: {
                [key: string]: number;
            };
        };
    };
};

const ResultsPage = () => {
    const {filename} = useContext(UploadContext);
    const [data, setData] = useState<Data>({});
    const [headers, setHeaders] = useState<string[]>([]);
    const [selectedKey, setSelectedKey] = useState<string>('');

    useEffect(() => {
        const fetchData = async () => {
            const response = await fetch(`http://localhost:5000/api/fetch/${filename}`);
            const jsonData = await response.json();
            setData(jsonData);
            setHeaders(Object.keys(jsonData));
            setSelectedKey(Object.keys(jsonData)[0]); // Select first outer key by default
        };

        if (filename) {
            fetchData();
        }
    }, [filename]);

    // Helper function to convert a string to a number
    const toNumber = (str: string): number => {
        const n = parseInt(str);
        return isNaN(n) ? 0 : n;
    };

    return (
        <div>
            <div className="border border-gray-200 rounded-md p-4 max-w-xs absolute top-8 right-8">
                <p className="text-lg font-semibold">
                    <a href="/" className="text-gray-800 no-underline hover:underline">
                        Main Page
                    </a>
                </p>
            </div>
            <div>
                <b>Select from the dropdown the Appropriate Attribute</b>
                <select value={selectedKey} onChange={(e) => setSelectedKey(e.target.value)}>
                    {headers.map((outerKey) => (
                        <option key={outerKey} value={outerKey}>
                            {outerKey}
                        </option>
                    ))}
                </select>
                {Object.keys(data).length > 0 && (
                    <table>
                        <thead>
                        <tr>
                            <th>System's Decision Making Confidence (%)</th>
                            <th>Occurrences</th>
                        </tr>
                        </thead>
                        <tbody>
                        {Object.keys(data[selectedKey]['outliers'])
                            .sort((a, b) => toNumber(a) - toNumber(b))
                            .map((innerKey, index, array) => {
                                const current = data[selectedKey]['outliers'][innerKey];
                                const previous = index > 0 ? data[selectedKey]['outliers'][array[index - 1]] : {};

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
                <h2 style={{fontSize: '60px', marginTop: '20px', marginBottom: '20px'}}>Patterns</h2>
                {Object.keys(data).length > 0 && (
                    <table>
                        <thead>
                        <tr>
                            <th>System's Decision Making Confidence (%)</th>
                            <th>Occurrences</th>
                        </tr>
                        </thead>
                        <tbody>
                        {Object.keys(data[selectedKey]['patterns']).map((innerKey, index, array) => {
                            const current = data[selectedKey]['patterns'][innerKey];
                            const previous = index > 0 ? data[selectedKey]['patterns'][array[index - 1]] : {};

                            const patterns: string[] = [];
                            for (const [key] of Object.entries(current)) {
                                if (!(key in previous)) {
                                    patterns.push(key);
                                }
                            }

                            if (patterns.length === 0) {
                                return null;
                            }

                            const threshold = toNumber(innerKey);

                            return (
                                <tr key={innerKey}>
                                    <td colSpan={1}
                                        style={{borderTop: '1px solid black', borderRight: '1px solid black'}}>
                                        {threshold}
                                    </td>
                                    <td colSpan={2} style={{borderTop: '1px solid black'}}>
                                        <table>
                                            <thead>
                                            <tr>
                                                <th>Patterns</th>
                                            </tr>
                                            </thead>
                                            <tbody>
                                            {patterns.map((pattern) => (
                                                <tr key={pattern}>
                                                    <td>{pattern}</td>
                                                </tr>
                                            ))}
                                            </tbody>
                                        </table>
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
        </div>
    );
};

export default ResultsPage;
