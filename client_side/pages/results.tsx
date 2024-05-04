import React, {useContext, useEffect, useState} from 'react';
import {UploadContext} from '../components/UploadContext';
import {Tab, Tabs} from "@mui/material";
import PatternsTable from "../components/PatternsTable";
import OutliersTable from "../components/OutliersTable";
import ToggleSwitch from "../components/ToggleSwitch";
import BriefSection from "../components/BriefSection";
import InfoBox from "../components/InfoBox";
import PageButton from "../components/PageButton";

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

    return (
        <div>
            <PageButton href={"/"} label={"Main Page"} icon={"🏠"} iconLabel={"home"}></PageButton>
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
            <ToggleSwitch onChange={(checked) => setDetailedView(checked)}/>
            {detailedView ? (
                <>
                    <OutliersTable resultsData={data} selectedKey={selectedKey}/>
                    <PatternsTable resultsData={data} selectedKey={selectedKey}/>
                </>
            ) : (
                data[selectedKey] ? (
                    <>
                        <BriefSection data={{[selectedKey]: {outliers: data[selectedKey]}}}
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
