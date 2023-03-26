import {useContext, useEffect, useState} from 'react';
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
            <b>Click on the Appropriate Attribute</b>
            <ul>
                {headers.map((outerKey) => (
                    <li key={outerKey}>
                        <button type="button" onClick={() => setSelectedKey(outerKey)}
                                className={outerKey === selectedKey ? 'active' : ''}>
                            {outerKey}
                        </button>
                    </li>
                ))}
            </ul>
            {Object.keys(data).length > 0 && (
                <table>
                    <thead>
                    <tr>
                        <th>Inner Key</th>
                        <th>Occurrences</th>
                    </tr>
                    </thead>
                    <tbody>
                    {Object.keys(data[selectedKey])
                        .sort((a, b) => toNumber(a) - toNumber(b)) // Sort innerKeys by numeric order
                        .map((innerKey) => (
                            <tr key={innerKey}>
                                <td colSpan={1} style={{
                                    borderTop: '1px solid black',
                                    borderRight: '1px solid black'
                                }}>{innerKey}</td>
                                <td colSpan={2}
                                    style={{borderTop: '1px solid black'}}>{JSON.stringify(data[selectedKey][innerKey])}</td>
                            </tr>
                        ))}
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
