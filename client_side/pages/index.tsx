import React, {useContext, useEffect, useRef, useState} from 'react';
import {useRouter} from 'next/router';
import {UploadContext} from '../components/UploadContext';
import Papa from 'papaparse';
import HistoryButton from "../components/HistoryButton";
import Footer from "../components/Footer";
import FancyTable from "../components/FancyTable";
import Header from "../components/Header";
import CircularProgress from '@mui/material/CircularProgress';

function FileUploadForm() {
    const {setFilename} = useContext(UploadContext);
    const fileInput = useRef<HTMLInputElement>(null);
    const numberInput = useRef<HTMLInputElement>(null);
    const longColumnCutoffInput = useRef<HTMLInputElement>(null)
    const largeFileThreshold = useRef<HTMLInputElement>(null)
    const router = useRouter();
    const [csvData, setCsvData] = useState<Array<Array<string>>>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState('');
    const [showAdvancedOptions, setShowAdvancedOptions] = useState(false);
    const [isStillLoading, setIsStillLoading] = useState(false);

    useEffect(() => {
        let timeout: NodeJS.Timeout | undefined;
        if (isLoading) {
            timeout = setTimeout(() => setIsStillLoading(true), 60000);
        } else {
            setIsStillLoading(false);
        }
        return () => clearTimeout(timeout);
    }, [isLoading]);

    const handleFileChange = () => {
        const file = fileInput.current?.files?.[0];
        if (!file) {
            return;
        }
        const fileSizeInMb = file.size / (1024 * 1024);
        const maxFileSizeInMb = 1;

        if (fileSizeInMb > maxFileSizeInMb) {
            setError('File is too large to be previewed on screen and will slow down your computer');
            return;
        }

        Papa.parse(file, {
            complete: (result) => {
                const data = result.data as string[][];
                setCsvData(data);
            },
        });
    };

    const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
        e.preventDefault();
        const formData = new FormData();
        const file = fileInput.current?.files?.[0];
        if (!file) {
            return;
        }
        formData.append('file', file);
        const number = numberInput.current?.value;
        formData.append('number', number || '');
        const long_column_cutoff = longColumnCutoffInput.current?.value
        formData.append('long_column_cutoff', long_column_cutoff || '')
        const largeFile_threshold_input = largeFileThreshold.current?.value
        formData.append('largeFile_threshold_input', largeFile_threshold_input || '')

        try {
            setIsLoading(true);
            const res = await fetch('http://localhost:5000/api/upload', {
                method: 'POST',
                body: formData,
                redirect: 'follow',
            });
            setFilename(file.name);
            if (res.redirected) {
                router.push(res.url);
            } else {
                const data = await res.json();
            }
        } catch (err) {
            console.error(err);
        } finally {
            setIsLoading(false);
        }
    };

    const toggleAdvancedOptions = () => {
        setShowAdvancedOptions((prev) => !prev);
    };

    return (
        <div className={'flex flex-col h-screen justify-between'}>
            <Header/>
            <p className="mb-6 text-lg font-normal text-gray-500 lg:text-xl sm:px-16 xl:px-48 dark:text-gray-400">
                Discover hidden insights and unlock the true potential of your data with our cutting-edge categorical
                outlier detection technology.
            </p>
            <div className="flex flex-col items-center justify-center">
                <form onSubmit={handleSubmit}>
                    <div className="mb-4">
                        <input type="file" ref={fileInput} onChange={handleFileChange}/>
                    </div>
                    <div className="mb-4">
                        {showAdvancedOptions && (
                            <div>
                                <div style={{display: "flex", alignItems: "center"}}>
                                    <span style={{marginRight: "10px"}}>Specify the ndistinct number:</span>
                                    <input type="number" ref={numberInput} placeholder="Enter a number"/>
                                </div>
                                <div style={{display: "flex", alignItems: "center"}}>
                                    <span style={{marginRight: "10px"}}>Specify the long column cutoff number:</span>
                                    <input type="number" ref={longColumnCutoffInput} placeholder="Enter a number"/>
                                </div>
                                <div style={{display: "flex", alignItems: "center"}}>
                                    <span style={{marginRight: "10px"}}>Specify above how many lines constitutes a large file:</span>
                                    <input type="number" ref={largeFileThreshold} placeholder="Enter a number"/>
                                </div>
                            </div>
                        )}
                    </div>
                    <button
                        type="submit"
                        className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-full"
                        disabled={isLoading}
                    >
                        Run
                    </button>
                </form>
                {error && <p className="text-red-500">{error}</p>}
                <button onClick={toggleAdvancedOptions}>
                    Click here to enable advanced options
                </button>
                {isLoading && (
                    <div className="flex items-center justify-center mt-4">
                        <p className="text-gray-500">FIONA is processing the dataset. Please wait...</p>
                        <CircularProgress/>
                    </div>
                )}
                {isStillLoading && (
                    <p className="text-blue-500 mt-4">Please do not worry, the process is still loading. Thank you for
                        your patience.</p>
                )}
            </div>
            <FancyTable csvData={csvData}/>
            <HistoryButton/>
            <Footer/>
        </div>
    );
}

export default FileUploadForm;