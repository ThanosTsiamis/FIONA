import React from 'react';

const HomeButton = () => (
    <div className="border border-gray-200 rounded-md p-4 max-w-xs absolute top-8 right-8">
        <p className="text-lg font-semibold">
            <a href="/" className="text-gray-800 no-underline hover:underline">
                Main Page
            </a>{' '}
            <span role="img" aria-label="house">
                🏠
            </span>
        </p>
    </div>
);

export default HomeButton;
