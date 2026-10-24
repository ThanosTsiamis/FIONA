import React, {useEffect, useState} from 'react';
import {Tab, Tabs} from "@mui/material";
import PatternsTable from "../components/PatternsTable";
import OutliersTable from "../components/OutliersTable";
import ToggleSwitch from "../components/ToggleSwitch";
import BriefSection from "../components/BriefSection";
import InfoBox from "../components/InfoBox";
import PageButton from "../components/PageButton";
import API_URL, {Results} from '../lib/api';

const HistoryPage = () => {
    // /api/history returns a plain array of file names.
    const [historyData, setHistoryData] = useState<string[]>([]);
    const [selectedFile, setSelectedFile] = useState<string>('');
    const [resultsData, setResultsData] = useState<Results>({});
    const [headers, setHeaders] = useState<string[]>([]);
    const [selectedKey, setSelectedKey] = useState<string>('');
    const [detailedView, setDetailedView] = useState<boolean>(false);
    const [loadError, setLoadError] = useState<string>('');

    useEffect(() => {
        const fetchHistoryData = async () => {
            try {
                const response = await fetch(`${API_URL}/api/history`);
                if (!response.ok) {
                    throw new Error(`Request failed with status ${response.status}`);
                }
                setHistoryData(await response.json());
            } catch (e) {
                console.error(e);
                setLoadError('Error fetching history from server. Are you sure the server is running?');
            }
        };
        fetchHistoryData();
    }, []);

    useEffect(() => {
        const fetchResultsData = async () => {
            try {
                const response = await fetch(`${API_URL}/api/fetch/${encodeURIComponent(selectedFile)}`);
                if (!response.ok) {
                    throw new Error(`Request failed with status ${response.status}`);
                }
                const jsonData: Results = await response.json();
                setResultsData(jsonData);
                const keys = Object.keys(jsonData);
                setHeaders(keys);
                setSelectedKey(keys[0] ?? ''); // Select first outer key by default
            } catch (e) {
                console.error(e);
                setLoadError('Error fetching results from server. Are you sure the server is running?');
            }
        };

        if (selectedFile) {
            setLoadError('');
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
                    {historyData.map((file) => (
                        <option key={file} value={file}>
                            {file}
                        </option>
                    ))}
                </select>
            </div>

            {loadError && <InfoBox message={loadError}/>}

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
                                <BriefSection data={{[selectedKey]: {outliers: resultsData[selectedKey].outliers}}}
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
