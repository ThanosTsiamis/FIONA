import React, {useEffect, useState} from 'react';
import {Tab, Tabs} from "@mui/material";
import PatternsTable from "../components/PatternsTable";
import OutliersTable from "../components/OutliersTable";
import ToggleSwitch from "../components/ToggleSwitch";
import BriefSection from "../components/BriefSection";
import InfoBox from "../components/InfoBox";
import PageButton from "../components/PageButton";
import {ApiError, fetchJson} from "../lib/api";

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
    const [historyData, setHistoryData] = useState<string[]>([]);
    const [selectedFile, setSelectedFile] = useState<string>('');
    const [resultsData, setResultsData] = useState<Data>({});
    const [headers, setHeaders] = useState<string[]>([]);
    const [selectedKey, setSelectedKey] = useState<string>('');
    const [detailedView, setDetailedView] = useState<boolean>(false);
    const [historyError, setHistoryError] = useState('');
    const [resultsError, setResultsError] = useState('');
    const [isHistoryLoading, setIsHistoryLoading] = useState(false);
    const [isResultsLoading, setIsResultsLoading] = useState(false);

    useEffect(() => {
        const fetchHistoryData = async () => {
            try {
                setIsHistoryLoading(true);
                setHistoryError('');
                const jsonData = await fetchJson<string[]>('/api/history');
                setHistoryData(jsonData);
            } catch (err) {
                if (err instanceof ApiError) {
                    setHistoryError(err.message);
                } else {
                    setHistoryError('Unable to fetch history from the server.');
                }
            } finally {
                setIsHistoryLoading(false);
            }
        };
        fetchHistoryData().then(r => r);
    }, []);

    useEffect(() => {
        const fetchResultsData = async () => {
            try {
                setIsResultsLoading(true);
                setResultsError('');
                const jsonData = await fetchJson<Data>(`/api/fetch/${selectedFile}`);
                setResultsData(jsonData);
                const nextHeaders = Object.keys(jsonData);
                setHeaders(nextHeaders);
                setSelectedKey(nextHeaders[0] || '');
            } catch (err) {
                if (err instanceof ApiError) {
                    setResultsError(err.message);
                } else {
                    setResultsError('Unable to fetch results from the server.');
                }
            } finally {
                setIsResultsLoading(false);
            }
        };

        if (selectedFile) {
            fetchResultsData().then(r => r);
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
                    {historyData.map((filename) => (
                        <option key={filename} value={filename}>
                            {filename}
                        </option>
                    ))}
                </select>
            </div>
            {isHistoryLoading && <p className="p-4 text-gray-600">Loading history...</p>}
            {historyError && <p className="p-4 text-red-500">{historyError}</p>}
            {!isHistoryLoading && !historyError && historyData.length === 0 && (
                <p className="p-4 text-gray-600">No previous result files are available yet.</p>
            )}

            {selectedFile && (
                <div>
                    {isResultsLoading && <p className="p-4 text-gray-600">Loading selected result...</p>}
                    {resultsError && <p className="p-4 text-red-500">{resultsError}</p>}
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
                    {headers.length > 0 && (
                        <ToggleSwitch onChange={(checked: boolean) => setDetailedView(checked)}/>
                    )}
                    {headers.length > 0 && detailedView ? (
                        <>
                            <OutliersTable resultsData={resultsData} selectedKey={selectedKey}/>
                            <PatternsTable resultsData={resultsData} selectedKey={selectedKey}/>
                        </>
                    ) : headers.length > 0 ? (
                        resultsData[selectedKey] ? (
                            <>
                                <BriefSection data={{[selectedKey]: {outliers: resultsData[selectedKey]}}}
                                              keyName={selectedKey}/>
                                <InfoBox
                                    message={"This bar chart orders bars from left to right to show certainty of outliers: the leftmost bar is the most significant outlier, and each subsequent bar to the right is less so."}/>
                            </>
                        ) : null
                    ) : null}
                </div>
            )}
        </div>
    );
};

export default HistoryPage;
