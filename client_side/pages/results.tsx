import React, {useContext, useEffect, useState} from 'react';
import {UploadContext} from '../components/UploadContext';

type Data = {
    [key: string]: {
        [key: string]: {
            [key: string]: number;
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
                        <th>Threshold Level</th>
                        <th>Occurrences</th>
                    </tr>
                    </thead>
                    <tbody>
                    {Object.keys(data[selectedKey])
                        .sort((a, b) => toNumber(a) - toNumber(b)) // Sort innerKeys by numeric order
                        .map((innerKey, index, array) => {
                            const current = data[selectedKey][innerKey];
                            const previous = index > 0 ? data[selectedKey][array[index - 1]] : {};

                            const occurrences: { [key: string]: number } = {};
                            for (const [key, value] of Object.entries(current)) {
                                if (!(key in previous)) {
                                    occurrences[key] = value;
                                }
                            }

                            if (Object.keys(occurrences).length === 0) {
                                return null;
                            }

                            return (
                                <tr key={innerKey}>
                                    <td colSpan={1} style={{
                                        borderTop: '1px solid black',
                                        borderRight: '1px solid black'
                                    }}>{innerKey}</td>
                                    <td colSpan={2}
                                        style={{borderTop: '1px solid black'}}>{JSON.stringify(occurrences)}</td>
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
    );
};

export default ResultsPage;
