import React from 'react';
import Tooltip from "@mui/material/Tooltip";

function Footer() {
    return (
        <footer className="bg-neutral-100 text-center dark:bg-neutral-600 lg:text-left">
            <div className="container p-1 text-neutral-800 dark:text-neutral-200">
                <div className="grid gap-4 lg:grid-cols-2">
                    <div className="mb-6 md:mb-0">
                        <h5 className="mb-2 font-medium uppercase">About</h5>
                        <p className="mb-4">
                            Fiona is the result of Thanos Tsiamis&apos; master thesis, developed under the
                            supervision of
                            Dr. A.A.A.
                            (Hakim) Qahtan for Utrecht University during the academic year 2022-2023.
                        </p>
                    </div>

                    <div className="grid gap-0.5 lg:grid-cols-2 mb-2 md:mb-0">
                        <h5 className="mb-2 font-medium uppercase">Links</h5>
                        <div className="mb-6 md:mb-0">
                            <ul className="mb-0 list-none">
                                <li>
                                    <a href="https://github.com/ThanosTsiamis/FIONA"
                                       className="text-neutral-800 dark:text-neutral-200">
                                        Github Repository
                                    </a>
                                </li>
                                <li>

                                    <a href="" className="text-neutral-800 dark:text-neutral-200">
                                        <Tooltip title="Coming soon!">
                                            <span>Master Thesis Paper</span>
                                        </Tooltip>
                                    </a>
                                </li>
                            </ul>
                        </div>
                        <div>
                            <ul className="mb-0 list-none">
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
            </div>
            <div
                className="bg-neutral-200 p-1 text-center text-neutral-700 dark:bg-neutral-700 dark:text-neutral-200">
                Developed at:
                <a className="text-neutral-800 dark:text-neutral-400" href="https://www.uu.nl/en/">
                    Utrecht University
                </a>
            </div>
        </footer>
    );
}

export default Footer;