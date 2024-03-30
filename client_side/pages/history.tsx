import React, {useEffect, useState} from 'react';
import {Tab, Tabs} from "@mui/material";
import HomeButton from "../components/HomeButton";
import PatternsTable from "../components/PatternsTable";
import OutliersTable from "../components/OutliersTable";
import ToggleSwitch from "../components/ToggleSwitch";

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

    return (
        <div>
            <HomeButton/>
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
                    <Tabs
                        value={selectedKey}
                        onChange={(e, newValue) => setSelectedKey(newValue)}
                        variant="scrollable"
                        scrollButtons="auto"
                        aria-label="tabs"
                    >
                        {headers.map((outerKey) => (
                            <Tab key={outerKey} value={outerKey} label={outerKey}/>
                        ))}
                    </Tabs>
                    <ToggleSwitch/>
                    <OutliersTable resultsData={resultsData} selectedKey={selectedKey}/>
                    <PatternsTable resultsData={resultsData} selectedKey={selectedKey}/>
                </div>
            )}
        </div>
    );
};

export default HistoryPage;
