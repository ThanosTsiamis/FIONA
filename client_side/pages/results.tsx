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
            <b>Select from the dropdown the Appropriate Attribute</b>
            <select value={selectedKey} onChange={(e) => setSelectedKey(e.target.value)}>
                {headers.map((outerKey) => (
                    <option key={outerKey} value={outerKey}>
                        {outerKey}
                    </option>
                ))}
            </select>
            <h2 style={{fontSize: '60px', marginTop: '20px', marginBottom: '20px'}}>Outliers</h2>
            {Object.keys(data).length > 0 && (
                <table>
                    <thead>
                    <tr>
                        <th>System's Decision Making Confidence (%)</th>
                        <th>Generalised Strings</th>
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
                                        style={{borderTop: '1px solid black', borderRight: '1px solid black'}}
                                        className={"text-center py-2"}>
                                        {threshold}
                                    </td>
                                    <td colSpan={1} style={{
                                        borderTop: '1px solid black',
                                        borderRight: '1px solid black',
                                        borderBottom: '1px solid black'
                                    }} className={"text-center py-2"}>
                                        {Object.keys(occurrences).map((key, i) => (
                                            <div key={key}>
                                                {key}
                                                {i !== Object.keys(occurrences).length - 1 && (
                                                    <hr className="border-dotted border-black my-1"/>
                                                )}
                                            </div>
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
            {data[selectedKey] && data[selectedKey]['patterns'] && (
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
                    {Object.keys(data[selectedKey]['patterns']).map((innerKey, index, array) => {
                        const current = data[selectedKey]['patterns'][innerKey];
                        const previous =
                            index < array.length - 1 ? data[selectedKey]['patterns'][array[index + 1]] : {};

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
    );
};

export default ResultsPage;
