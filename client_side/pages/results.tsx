import React, {useContext, useEffect, useState} from 'react';
import {UploadContext} from '../components/UploadContext';
import {Tab, Tabs} from "@mui/material";
import PatternsTable from "../components/PatternsTable";
import OutliersTable from "../components/OutliersTable";
import ToggleSwitch from "../components/ToggleSwitch";
import BriefSection from "../components/BriefSection";
import InfoBox from "../components/InfoBox";
import PageButton from "../components/PageButton";
import API_URL, {Results} from '../lib/api';

const ResultsPage = () => {
    const {filename, setFilename} = useContext(UploadContext);
    const [data, setData] = useState<Results>({});
    const [headers, setHeaders] = useState<string[]>([]);
    const [selectedKey, setSelectedKey] = useState<string>('');
    const [detailedView, setDetailedView] = useState<boolean>(false);
    const [loadError, setLoadError] = useState<string>('');

    useEffect(() => {
        const fetchData = async (name: string) => {
            try {
                const response = await fetch(`${API_URL}/api/fetch/${encodeURIComponent(name)}`);
                if (!response.ok) {
                    throw new Error(`Request failed with status ${response.status}`);
                }
                const jsonData: Results = await response.json();
                setData(jsonData);
                const keys = Object.keys(jsonData);
                setHeaders(keys);
                setSelectedKey(keys[0] ?? ''); // Select first outer key by default
            } catch (e) {
                console.error(e);
                setLoadError('Could not load these results. Is the API server running?');
            }
        };

        if (filename) {
            setLoadError('');
            fetchData(filename);
            return;
        }

        // The upload context is in-memory, so a refresh (or a shared link) leaves us
        // with no filename and an empty page. Fall back to the most recent result.
        const loadMostRecent = async () => {
            try {
                const response = await fetch(`${API_URL}/api/history`);
                if (!response.ok) {
                    throw new Error(`Request failed with status ${response.status}`);
                }
                const files: string[] = await response.json();
                if (files.length === 0) {
                    throw new Error('no results');
                }
                // os.listdir order is arbitrary, so sort for deterministic behaviour.
                setFilename(files.sort().reverse()[0]);
            } catch (e) {
                console.error(e);
                setLoadError('No dataset loaded. Upload one from the main page, or pick a previous run in History.');
            }
        };

        loadMostRecent();
    }, [filename, setFilename]);

    return (
        <div>
            <PageButton href={"/"} label={"Main Page"} icon={"🏠"} iconLabel={"home"}></PageButton>
            {headers.length > 0 && (
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
            )}
            <ToggleSwitch onChange={(checked) => setDetailedView(checked)}/>
            {loadError && <InfoBox message={loadError}/>}
            {detailedView ? (
                <>
                    <OutliersTable resultsData={data} selectedKey={selectedKey}/>
                    <PatternsTable resultsData={data} selectedKey={selectedKey}/>
                </>
            ) : (
                data[selectedKey] ? (
                    <>
                        <BriefSection data={{[selectedKey]: {outliers: data[selectedKey].outliers}}}
                                      keyName={selectedKey}/>
                        <InfoBox
                            message={"This bar chart orders bars from left to right to show certainty of outliers: the leftmost bar is the most significant outlier, and each subsequent bar to the right is less so."}/>
                    </>
                ) : null
            )}
        </div>
    );
};

export default ResultsPage;
