import React, {useContext, useEffect, useState} from 'react';
import {UploadContext} from '../components/UploadContext';
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

const ResultsPage = () => {
    const {filename} = useContext(UploadContext);
    const [data, setData] = useState<Data>({});
    const [headers, setHeaders] = useState<string[]>([]);
    const [selectedKey, setSelectedKey] = useState<string>('');
    const [detailedView, setDetailedView] = useState<boolean>(false);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState('');

    useEffect(() => {
        const fetchData = async () => {
            try {
                setIsLoading(true);
                setError('');
                const jsonData = await fetchJson<Data>(`/api/fetch/${filename}`);
                const nextHeaders = Object.keys(jsonData);
                setData(jsonData);
                setHeaders(nextHeaders);
                setSelectedKey(nextHeaders[0] || '');
            } catch (err) {
                if (err instanceof ApiError) {
                    setError(err.message);
                } else {
                    setError('Unable to load results. Please try again.');
                }
            } finally {
                setIsLoading(false);
            }
        };

        if (filename) {
            fetchData().then(r => r);
        }
    }, [filename]);

    return (
        <div>
            <PageButton href={"/"} label={"Main Page"} icon={"🏠"} iconLabel={"home"}></PageButton>
            {!filename && <p className="p-4 text-gray-600">No uploaded file is selected yet.</p>}
            {isLoading && <p className="p-4 text-gray-600">Loading results...</p>}
            {error && <p className="p-4 text-red-500">{error}</p>}
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
            {headers.length > 0 && <ToggleSwitch onChange={(checked) => setDetailedView(checked)}/>}
            {headers.length > 0 && detailedView ? (
                <>
                    <OutliersTable resultsData={data} selectedKey={selectedKey}/>
                    <PatternsTable resultsData={data} selectedKey={selectedKey}/>
                </>
            ) : headers.length > 0 ? (
                data[selectedKey] ? (
                    <>
                        <BriefSection data={{[selectedKey]: {outliers: data[selectedKey]}}}
                                      keyName={selectedKey}/>
                        <InfoBox
                            message={"This bar chart orders bars from left to right to show certainty of outliers: the leftmost bar is the most significant outlier, and each subsequent bar to the right is less so."}/>
                    </>
                ) : null
            ) : null}
        </div>
    );
};

export default ResultsPage;
