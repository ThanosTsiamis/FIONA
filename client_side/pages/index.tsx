import React, {useContext, useRef, useState} from 'react';
import {useRouter} from 'next/router';
import {UploadContext} from '../components/UploadContext';
import Papa from 'papaparse';

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

    return (
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
    );
}

export default FileUploadForm;
