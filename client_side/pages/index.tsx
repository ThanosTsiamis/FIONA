import React, {useContext, useRef, useState} from 'react';
import {useRouter} from 'next/router';
import {UploadContext} from '../components/UploadContext';
import Papa from 'papaparse';
import Image from "next/image";

function FileUploadForm() {
    const {filename, setFilename} = useContext(UploadContext);
    const fileInput = useRef<HTMLInputElement>(null);
    const router = useRouter();
    const [csvData, setCsvData] = useState<Array<Array<string>>>([]);

    const handleFileChange = () => {
        const file = fileInput.current?.files?.[0];
        if (!file) {
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

        try {
            const res = await fetch('http:localhost:5000/api/upload', {
                method: 'POST',
                body: formData,
            });
            setFilename(file.name);
            // Navigate to the results page
            router.push('/results');
        } catch (err) {
            console.error(err);
        }
    };

    return (<div>
            <h1 className="mb-4 ml-10 text-4xl font-extrabold leading-none tracking-tight text-gray-900 md:text-1xl lg:text-5xl dark:text-white">Outlier
                Detector</h1>
            <p className="mb-6 text-lg font-normal text-gray-500 lg:text-xl sm:px-16 xl:px-48 dark:text-gray-400">Discover
                hidden insights and unlock the true potential of your data with our cutting-edge categorical outlier
                detection technology.</p>

            <form onSubmit={handleSubmit}>
                <input type="file" ref={fileInput} onChange={handleFileChange}/>
                <button type="submit">Upload</button>
                <table>
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
            </form>
            <footer className="fixed inset-x-0 bottom-0">
                <div className="sm:items-center sm:justify-between">
                    <a href="https://www.uu.nl/en/" className="flex items-center mb-4 sm:mb-0">
                        <Image src="/UU_logo_2021_EN_RGB.png"
                               alt="Utrecht University Logo" width="158" height={64}/>
                        <span
                            className="self-center text-base whitespace-nowrap dark:text-white">Utrecht University</span>
                    </a>
                </div>

            </footer>
        </div>
    );
}

export default FileUploadForm;
