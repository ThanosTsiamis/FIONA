import type {NextPage} from "next";
import Image from "next/image";
import Papa from "papaparse";
import {useState} from "react";
import {useRouter} from 'next/router';

const Home: NextPage = () => {

    // State to store parsed data
    const [parsedData, setParsedData] = useState<any[]>([]);
    //State to store table Column name
    const [tableRows, setTableRows] = useState<string[]>([]);
    //State to store the values
    const [values, setValues] = useState<any[][]>([]);

    const changeHandler = (event: React.ChangeEvent<HTMLInputElement>) => {
        // Passing file data (event.target.files[0]) to parse using Papa.parse
        Papa.parse(event.target.files![0], {
            header: true,
            skipEmptyLines: true,
            complete: function (results) {
                const rowsArray: string[][] = [];
                const valuesArray: any[][] = [];

                // Iterating data to get column name and their values
                results.data.map((d: any) => {
                    rowsArray.push(Object.keys(d));
                    valuesArray.push(Object.values(d));
                });

                // Parsed Data Response in array format
                setParsedData(results.data);

                // Filtered Column Names
                setTableRows(rowsArray[0]);

                // Filtered Values
                setValues(valuesArray);
            },
        });
    };

    const router = useRouter();
    const handleButtonClick = () => {
        router.push("/pages/results");
    };

    return (
        <div>
            <h1 className="mb-4 ml-10 text-4xl font-extrabold leading-none tracking-tight text-gray-900 md:text-1xl lg:text-5xl dark:text-white">Outlier
                Detector</h1>
            <p className="mb-6 text-lg font-normal text-gray-500 lg:text-xl sm:px-16 xl:px-48 dark:text-gray-400">Discover
                hidden insights and unlock the true potential of your data with our cutting-edge categorical outlier
                detection technology.</p>

            <main className="py-10">
                <div className="w-full max-w-3xl px-3 mx-auto">
                    <h1 className="mb-10 text-3xl font-bold text-gray-900">
                        Provide a file
                    </h1>

                    <div className="space-y-10">
                        <div>
                            <form action="http://localhost:5000/api/upload" method="POST" encType="multipart/form-data">
                                <input type="file" name="file" onChange={changeHandler}/>
                                <button
                                    className={"bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"}
                                    type="submit" onClick={handleButtonClick}>Submit file
                                </button>
                            </form>
                        </div>
                    </div>
                </div>
            </main>

            <table>
                <thead>
                <tr>
                    {tableRows.map((rows: string, index: number) => {
                        return <th key={index}>{rows}</th>;
                    })}
                </tr>
                </thead>
                <tbody>
                {values.map((value: any[], index: number) => {
                    return (
                        <tr key={index}>
                            {value.map((val: any, i: number) => {
                                return <td key={i}>{val}</td>;
                            })}
                        </tr>
                    );
                })}
                </tbody>
            </table>
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
};

export default Home;
