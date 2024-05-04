import React, {useEffect, useState} from 'react';
import {Tab, Tabs} from "@mui/material";
import PatternsTable from "../components/PatternsTable";
import OutliersTable from "../components/OutliersTable";
import ToggleSwitch from "../components/ToggleSwitch";
import BriefSection from "../components/BriefSection";
import InfoBox from "../components/InfoBox";
import PageButton from "../components/PageButton";

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
    const [detailedView, setDetailedView] = useState<boolean>(false);

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
            <PageButton href={"/"} label={"Main Page"} icon={"🏠"} iconLabel={"home"}></PageButton>
            <div className="bg-white shadow-md rounded-lg p-4 max-w-sm mx-auto ml-4">
                <label htmlFor="fileSelect" className="block text-sm font-medium text-gray-700 mb-1">
                    Select the JSON file:
                </label>
                <select
                    id="fileSelect"
                    value={selectedFile}
                    onChange={(e) => setSelectedFile(e.target.value)}
                    className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md shadow-sm"
                >
                    <option value="">-- Select a file --</option>
                    {Object.keys(historyData).map((key) => (
                        <option key={key} value={historyData[key]}>
                            {historyData[key]}
                        </option>
                    ))}
                </select>
            </div>

            {selectedFile && (
                <div>
                    <Tabs
                        value={selectedKey}
                        onChange={(e, newValue) => setSelectedKey(newValue)}
                        variant="scrollable"
                        scrollButtons="auto"
                        aria-label="tabs"
                    >
                        {headers.map((headerKey) => (
                            <Tab key={headerKey} value={headerKey} label={headerKey}/>
                        ))}
                    </Tabs>
                    <ToggleSwitch
                        onChange={(checked: boolean) => setDetailedView(checked)}/>
                    {detailedView ? (
                        <>
                            <OutliersTable resultsData={resultsData} selectedKey={selectedKey}/>
                            <PatternsTable resultsData={resultsData} selectedKey={selectedKey}/>
                        </>
                    ) : (
                        resultsData[selectedKey] ? (
                            <>
                                <BriefSection data={{[selectedKey]: {outliers: resultsData[selectedKey]}}}
                                              keyName={selectedKey}/>
                                <InfoBox
                                    message={"This bar chart orders bars from left to right to show certainty of outliers: the leftmost bar is the most significant outlier, and each subsequent bar to the right is less so."}/>
                            </>
                        ) : null
                    )}
                </div>
            )}
        </div>
    );
};

export default HistoryPage;
