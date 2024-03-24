import React, {useContext, useRef, useState} from 'react';
import {useRouter} from 'next/router';
import {UploadContext} from '../components/UploadContext';
import Papa from 'papaparse';

function FileUploadForm() {
    const {filename, setFilename} = useContext(UploadContext);
    const fileInput = useRef<HTMLInputElement>(null);
    const numberInput = useRef<HTMLInputElement>(null);
    const longColumnCutoffInput = useRef<HTMLInputElement>(null)
    const largeFileThreshold = useRef<HTMLInputElement>(null)
    const router = useRouter();
    const [csvData, setCsvData] = useState<Array<Array<string>>>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [enableParallel, setEnableParallel] = useState(false);
    const [error, setError] = useState('');
    const [showAdvancedOptions, setShowAdvancedOptions] = useState(false); // State variable to control visibility

    const handleFileChange = () => {
        const file = fileInput.current?.files?.[0];
        if (!file) {
            return;
        }
        const fileSizeInMb = file.size / (1024 * 1024);
        const maxFileSizeInMb = 1;

        if (fileSizeInMb > maxFileSizeInMb) {
            setError(
                'File is too large to be previewed on screen and will slow down your computer'
            );
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

        // Add the number value to the form data
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
            <header className="flex items-center mb-4 ml-10 bg-gray-100">
                <img src="/LogoFIONA.png" alt="Logo" className="w-48 h-48 mr-2 "/>
                <h1 className="text-4xl font-extrabold leading-none tracking-tight text-gray-900 md:text-1xl lg:text-5xl dark:text-white">
                    FIONA: Categorical Outlier Detector
                </h1>
            </header>
            <p className="mb-6 text-lg font-normal text-gray-500 lg:text-xl sm:px-16 xl:px-48 dark:text-gray-400">
                Discover hidden insights and unlock the true potential of your data with our cutting-edge categorical
                outlier
                detection technology.
            </p>
            <div className="flex flex-col items-center justify-center">
                <form onSubmit={handleSubmit}>
                    {/* File input */}
                    <div className="mb-4">
                        <input type="file" ref={fileInput} onChange={handleFileChange}/>
                    </div>
                    {/* Number input */}
                    <div className="mb-4">
                        {/* Show the advanced options input based on the state */}
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
                        Upload
                    </button>
                </form>
                {error && <p className="text-red-500">{error}</p>} {/* Display error message */}
                <button onClick={toggleAdvancedOptions}>
                    {/* Toggle the label based on the state */}
                    Click here to enable advanced options
                </button>
                {isLoading && (
                    <div className="flex items-center justify-center mt-4">
                        <div
                            className="inline-block h-8 w-8 animate-spin rounded-full border-4 border-solid border-current border-r-transparent align-[-0.125em] motion-reduce:animate-[spin_1.5s_linear_infinite]"
                            role="status"
                        >
<span className="!absolute !-m-px !h-px !w-px !overflow-hidden !whitespace-nowrap !border-0 !p-0 ![clip:rect(0,0,0,0)]">
Processing...
</span>
                        </div>
                    </div>
                )}
            </div>
            <table className="mt-4">
                <thead>
                <tr>
                    {csvData[0]?.map((header) => (
                        <th key={header}>{header}</th>
                    ))}
                </tr>
                </thead>
                <tbody>
                {csvData.slice(1).map((row, index) => (
                    <tr key={index}>
                        {row.map((cell, index) => (
                            <td key={index}>{cell}</td>
                        ))}
                    </tr>
                ))}
                </tbody>
            </table>
            <div className="border border-gray-200 rounded-md p-4 max-w-xs absolute top-8 right-8">
                <p className="text-lg font-semibold">
                    <a href="history" className="text-gray-800 no-underline hover:underline">
                        History
                    </a>{' '}
                    <span role="img" aria-label="book">
📖
</span>
                </p>
            </div>
            <footer className="bg-neutral-100 text-center dark:bg-neutral-600 lg:text-left">
                <div className="container p-6 text-neutral-800 dark:text-neutral-200">
                    <div className="grid gap-4 lg:grid-cols-2">
                        <div className="mb-6 md:mb-0">
                            <h5 className="mb-2 font-medium uppercase">About</h5>
                            <p className="mb-4">
                                Fiona is the result of Thanos Tsiamis&apos; master thesis, developed under the supervision of
                                Dr. A.A.A.
                                (Hakim) Qahtan for Utrecht University during the academic year 2022-2023.
                            </p>
                        </div>

                        <div className="mb-6 md:mb-0">
                            <h5 className="mb-2 font-medium uppercase">Links</h5>

                            <ul className="mb-0 list-none">
                                <li>
                                    <a href="" className="text-neutral-800 dark:text-neutral-200">
                                        Github Repository
                                    </a>
                                </li>
                                <li>
                                    <a href="" className="text-neutral-800 dark:text-neutral-200">
                                        Master Thesis Paper (coming soon)
                                    </a>
                                </li>
                                <li>
                                    <a href="https://github.com/ThanosTsiamis"
                                       className="text-neutral-800 dark:text-neutral-200">
                                        Thanos Tsiamis&apos;s Github
                                    </a>
                                </li>
                                <li>
                                    <a href="https://github.com/qahtanaa"
                                       className="text-neutral-800 dark:text-neutral-200">
                                        Dr. A.A.A. (Hakim) Qahtan&apos;s Github
                                    </a>
                                </li>
                            </ul>
                        </div>
                    </div>
                </div>
                <div
                    className="bg-neutral-200 p-4 text-center text-neutral-700 dark:bg-neutral-700 dark:text-neutral-200">
                    Developed at:
                    <a className="text-neutral-800 dark:text-neutral-400" href="https://www.uu.nl/en/">
                        Utrecht University
                    </a>
                </div>
            </footer>
        </div>
    );
}

export default FileUploadForm;