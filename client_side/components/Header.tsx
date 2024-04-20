import React, {useState} from 'react';
import Image from "next/image";

function Header() {
    const [hover,setHover] = useState(false)

    return (
        <header className="flex items-center mb-4 ml-10 bg-gray-100">
            <div onMouseEnter={() => setHover(true)} onMouseLeave={() => setHover(false)}>
                <Image
                    src={hover ? "/LogoFIONA_Green_Leaves.png" : "/LogoFIONA.png"}
                    alt="Logo of FIONA"
                    width={126}
                    height={126}
                    className="w-48 h-48 mr-2 "
                />
            </div>
            <h1 className="text-4xl font-extrabold leading-none tracking-tight text-gray-900 md:text-1xl lg:text-5xl dark:text-white">
                FIONA: Categorical Outlier Detector
            </h1>
        </header>
    );
}

export default Header;