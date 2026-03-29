import React, {useContext, useEffect, useRef, useState} from 'react';
import {useRouter} from 'next/router';
import {UploadContext} from '../components/UploadContext';
import Papa from 'papaparse';
import Footer from "../components/Footer";
import FancyTable from "../components/FancyTable";
import Header from "../components/Header";
import CircularProgress from '@mui/material/CircularProgress';
import PageButton from "../components/PageButton";
import {ApiError, uploadDataset} from "../lib/api";

const PREVIEW_SIZE_LIMIT_MB = 1;
const SUPPORTED_FILE_EXTENSIONS = ['csv', 'xlsx', 'json', 'tsv'];

function getFileExtension(filename: string) {
    return filename.split('.').pop()?.toLowerCase() || '';
}

function isPositiveInteger(value: string) {
    if (!value) {
        return true;
    }

    return /^\d+$/.test(value) && Number(value) > 0;
}

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
        if (!file) {
            setCsvData([]);
            return;
        }

        setError('');
        const fileExtension = getFileExtension(file.name);
        if (!SUPPORTED_FILE_EXTENSIONS.includes(fileExtension)) {
            setCsvData([]);
            setError('Unsupported file type. Please upload a CSV, XLSX, JSON, or TSV file.');
            return;
        }

        const fileSizeInMb = file.size / (1024 * 1024);
        if (fileSizeInMb > PREVIEW_SIZE_LIMIT_MB) {
            setCsvData([]);
            setError('File is too large to be previewed on screen and will slow down your computer');
            return;
        }

        if (fileExtension !== 'csv' && fileExtension !== 'tsv') {
            setCsvData([]);
            return;
        }

        Papa.parse(file, {
            complete: (result) => {
                const data = result.data as string[][];
                setCsvData(data);
            },
            error: () => {
                setCsvData([]);
                setError('Unable to preview the selected file.');
            }
        });
    };

    const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
        e.preventDefault();
        const file = fileInput.current?.files?.[0];
        if (!file) {
            setError('Please choose a file before running the algorithm.');
            return;
        }

        const number = numberInput.current?.value || '';
        const long_column_cutoff = longColumnCutoffInput.current?.value || '';
        const largeFile_threshold_input = largeFileThreshold.current?.value || '';

        if (!isPositiveInteger(number)) {
            setError('The ndistinct number must be a positive integer.');
            return;
        }

        if (!isPositiveInteger(long_column_cutoff)) {
            setError('The long column cutoff must be a positive integer.');
            return;
        }

        if (!isPositiveInteger(largeFile_threshold_input)) {
            setError('The large file threshold must be a positive integer.');
            return;
        }

        const formData = new FormData();
        formData.append('file', file);
        formData.append('number', number);
        formData.append('long_column_cutoff', long_column_cutoff);
        formData.append('largeFile_threshold_input', largeFile_threshold_input);

        formData.append('regex_transformation_only', regexCheckBox.toString() || '');
        formData.append('generalised_transformation_only', generalisedCheckBox.toString() || '');

        try {
            setError('');
            setIsLoading(true);
            const res = await uploadDataset(formData);
            setFilename(file.name);
            if (res.redirected) {
                router.push(res.url);
            } else {
                router.push('/results');
            }
        } catch (err) {
            if (err instanceof ApiError) {
                setError(err.message);
            } else {
                setError('The upload failed. Please check that the backend is running and try again.');
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
                            onChange={handleFileChange}
                            accept=".csv,.xlsx,.json,.tsv"
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
                        className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-full"
                        disabled={isLoading}
                    >
                        Run
                    </button>
                </form>
                {error && <p className="text-red-500">{error}</p>}
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
