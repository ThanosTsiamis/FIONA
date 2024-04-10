import React from 'react';

interface InfoBoxProps {
    message: string;
}

const InfoBox: React.FC<InfoBoxProps> = ({message}) => {
    return (
        <div className="info-box bg-blue-100 border-l-4 border-blue-500 text-blue-700 p-4 mb-4" role="alert">
            <p className="font-bold">Info</p>
            <p>{message}</p>
        </div>
    );
};

export default InfoBox;
