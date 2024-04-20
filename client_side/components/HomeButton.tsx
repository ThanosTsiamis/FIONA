import React from 'react';
import Link from "next/link";

const HomeButton = () => (
    <div className="border border-gray-200 rounded-md p-4 max-w-xs absolute top-8 right-8">
        <p className="text-lg font-semibold">
            <Link href="/" className="text-gray-800 no-underline hover:underline">
                Main Page
            </Link>
            <span role="img" aria-label="house">
                🏠
            </span>
        </p>
    </div>
);

export default HomeButton;
