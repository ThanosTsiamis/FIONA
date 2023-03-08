import React, {useContext, useEffect, useState} from "react";
import {UploadContext} from "../components/UploadContext";

function ResultsPage() {
    const {filename} = useContext(UploadContext);
    const [data, setData] = useState<any>(null);

    useEffect(() => {
        const fetchData = async () => {
            const response = await fetch(`http://localhost:5000/api/fetch/${filename}`);
            const jsonData = await response.json();
            setData(jsonData);
        };

        if (filename) {
            fetchData();
        }
    }, [filename]);

    return (
        <div>
            {data ? (
                <div>
                    <h2>Results for {filename}:</h2>
                    <pre>{JSON.stringify(data, null, 2)}</pre>
                </div>
            ) : (
                <p>Loading...</p>
            )}
        </div>
    );
}

export default ResultsPage;
