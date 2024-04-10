import React from 'react';
import Image from "next/image";

function Header() {
    return (
        <header className="flex items-center mb-4 ml-10 bg-gray-100">
            <Image src="/LogoFIONA.png" alt="Logo of FIONA" width={126} height={126} className="w-48 h-48 mr-2 "/>
            <h1 className="text-4xl font-extrabold leading-none tracking-tight text-gray-900 md:text-1xl lg:text-5xl dark:text-white">
                FIONA: Categorical Outlier Detector
            </h1>
        </header>
    );
}

export default Header;