import React from 'react';

const HistoryButton = () => (
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
);

export default HistoryButton;