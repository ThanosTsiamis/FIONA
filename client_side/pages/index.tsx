import React, {useContext, useEffect, useRef, useState} from 'react';
import {useRouter} from 'next/router';
import {UploadContext} from '../components/UploadContext';
import Papa from 'papaparse';
import Footer from "../components/Footer";
import FancyTable from "../components/FancyTable";
import Header from "../components/Header";
import CircularProgress from '@mui/material/CircularProgress';
import PageButton from "../components/PageButton";
import API_URL, {isPreviewableDataset, isSupportedUpload, readApiError} from '../lib/api';

function FileUploadForm() {
    const {setFilename} = useContext(UploadContext);
    const fileInput = useRef<HTMLInputElement>(null);
    const numberInput = useRef<HTMLInputElement>(null);
    const longColumnCutoffInput = useRef<HTMLInputElement>(null)
    const largeFileThreshold = useRef<HTMLInputElement>(null)
    const router = useRouter();
    const [csvData, setCsvData] = useState<Array<Array<string>>>([]);
    const [hasFile, setHasFile] = useState(false);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState('');
    const [previewNote, setPreviewNote] = useState('');
    const [showAdvancedOptions, setShowAdvancedOptions] = useState(false);
    const [isStillLoading, setIsStillLoading] = useState(false);
    const [regexCheckBox, setRegexCheckbox] = useState(false);
    const [generalisedCheckBox, setGeneralisedCheckBox] = useState(false);

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
        setError('');
        setPreviewNote('');
        setCsvData([]);
        setHasFile(Boolean(file));
        if (!file) {
            return;
        }

        if (!isSupportedUpload(file.name)) {
            setError('Unsupported file format. Use CSV, XLSX, JSON, or TSV.');
            return;
        }

        if (!isPreviewableDataset(file.name)) {
            setPreviewNote('Preview is only available for CSV and TSV files. You can still run detection on this file.');
            return;
        }

        const fileSizeInMb = file.size / (1024 * 1024);
        const maxFileSizeInMb = 1;

        if (fileSizeInMb > maxFileSizeInMb) {
            setPreviewNote('File is too large to preview here. You can still run detection on it.');
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
            setError('Choose a dataset file first.');
            return;
        }
        if (!isSupportedUpload(file.name)) {
            setError('Unsupported file format. Use CSV, XLSX, JSON, or TSV.');
            return;
        }
        formData.append('file', file);
        const number = numberInput.current?.value;
        formData.append('number', number || '');
        const long_column_cutoff = longColumnCutoffInput.current?.value
        formData.append('long_column_cutoff', long_column_cutoff || '')
        const largeFile_threshold_input = largeFileThreshold.current?.value
        formData.append('largeFile_threshold_input', largeFile_threshold_input || '')

        formData.append('regex_transformation_only', regexCheckBox ? 'true' : '');
        formData.append('generalised_transformation_only', generalisedCheckBox ? 'true' : '');

        try {
            setIsLoading(true);
            setError('');
            const res = await fetch(`${API_URL}/api/upload`, {
                method: 'POST',
                body: formData,
            });
            if (!res.ok) {
                throw new Error(await readApiError(res, 'Upload failed'));
            }
            const {result}: {result: string} = await res.json();
            setFilename(result);
            await router.push('/results');
        } catch (err) {
            console.error(err);
            // fetch raises TypeError when the API is unreachable (Chrome: "Failed to fetch").
            if (err instanceof TypeError) {
                setError('The upload failed. Is the API server running on ' + API_URL + '?');
            } else {
                setError(err instanceof Error ? err.message : 'The upload failed.');
            }
        } finally {
            setIsLoading(false);
        }
    };

    const toggleAdvancedOptions = () => {
        setShowAdvancedOptions((prev) => !prev);
    };

    const handleCheckbox1Change = () => {
        setRegexCheckbox(!regexCheckBox);
        if (!regexCheckBox) {
            setGeneralisedCheckBox(false);
        }
    };

    const handleCheckbox2Change = () => {
        setGeneralisedCheckBox(!generalisedCheckBox);
        if (!generalisedCheckBox) {
            setRegexCheckbox(false);
        }
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
                        <input
                            type="file"
                            ref={fileInput}
                            accept=".csv,.xlsx,.json,.tsv"
                            onChange={handleFileChange}
                        />
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
                                <div style={{display: "flex", alignItems: "center"}}>
                                    <label style={{marginRight: "10px"}}>
                                        <input
                                            type="checkbox"
                                            checked={regexCheckBox}
                                            onChange={handleCheckbox1Change}
                                        />
                                        Regex transformations only
                                    </label>
                                    <label style={{marginRight: "10px"}}>
                                        <input
                                            type="checkbox"
                                            checked={generalisedCheckBox}
                                            onChange={handleCheckbox2Change}
                                        />
                                        Generalised transformations only
                                    </label>
                                </div>
                            </div>
                        )}
                    </div>
                    <button
                        type="submit"
                        className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-full disabled:bg-gray-400 disabled:cursor-not-allowed"
                        disabled={isLoading || !hasFile}
                    >
                        Run
                    </button>
                </form>
                {error && <p className="text-red-500">{error}</p>}
                {previewNote && <p className="text-gray-500">{previewNote}</p>}
                <button
                    onClick={toggleAdvancedOptions}
                    disabled={showAdvancedOptions}
                    className={`mt-4 rounded-full py-3 pr-5 ${showAdvancedOptions ? 'bg-gray-300 text-gray-500 cursor-not-allowed' : 'bg-gray-50 hover:bg-gray-200 text-black'}`}
                >
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
            <PageButton href={"history"} label={"History"} icon={"📖"} iconLabel={"book"}></PageButton>
            <Footer/>
        </div>
    );
}

export default FileUploadForm;
