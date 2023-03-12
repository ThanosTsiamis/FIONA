import React, {useContext, useEffect, useState} from "react";
import {UploadContext} from "../components/UploadContext";

function ResultsPage() {
    const {filename} = useContext(UploadContext);
    const [data, setData] = useState<any>(null);

    useEffect(() => {
        const fetchData = async () => {
            const response = await fetch(
                `http://localhost:5000/api/fetch/${filename}`
            );
            const jsonData = await response.json();
            setData(jsonData);
        };

        if (filename) {
            fetchData();
        }
    }, [filename]);

    const renderTable = (obj: any) => {
        const headers = Object.keys(obj);
        const rows = Object.values(obj).map((row: any, index: number) => {
            return (
                <tr key={index}>
                    {headers.map((header, index) => (
                        <td key={index}>{JSON.stringify(row[header])}</td>
                    ))}
                </tr>
            );
        });

        return (
            <table>
                <thead>
                <tr>
                    {headers.map((header, index) => (
                        <th key={index}>{header}</th>
                    ))}
                </tr>
                </thead>
                <tbody>{rows}</tbody>
            </table>
        );
    };

    return (
        <div>
            {data ? (
                <div>
                    <h2>Results for {filename}:</h2>
                    {data.map((obj: any, index: number) => (
                        <div key={index}>{renderTable(obj)}</div>
                    ))}
                </div>
            ) : (
                <p>Loading...</p>
            )}
        </div>
    );
}

export default ResultsPage;
