import React from 'react';
import {Bar} from 'react-chartjs-2';
import {
    BarElement,
    CategoryScale,
    Chart as ChartJS,
    Legend,
    LinearScale,
    LogarithmicScale,
    Title,
    Tooltip
} from 'chart.js';
import {OutliersByThreshold, Results} from '../lib/api';

ChartJS.register(
    CategoryScale,
    LinearScale,
    BarElement,
    Title,
    Tooltip,
    Legend,
    LogarithmicScale
);

const processData = (outliers: OutliersByThreshold) => {
    const labels = new Set<string>();
    const dataPoints: { [key: string]: number } = {};
    const seenOutliers: { [key: string]: Set<string> } = {};
    // `outliers` is the outlier dictionary itself: {threshold: {representative: {value: count}}}
    const dataItem = outliers;

    Object.entries(dataItem).forEach(([band, representatives]) => {
        Object.entries(representatives).forEach(([representative, outlierwithFreq]) => {
            const label = `${representative}`;
            labels.add(label);
            seenOutliers[label] = seenOutliers[label] || new Set<string>();
            Object.entries(outlierwithFreq).forEach(([outlier, freq]) => {
                if (!seenOutliers[label].has(outlier)) {
                    dataPoints[label] = (dataPoints[label] || 0) + freq;
                    seenOutliers[label].add(outlier);
                }
            });
        });
    });


    return {
        labels: Array.from(labels),
        datasets: [
            {
                label: `Outliers`,
                data: Array.from(labels).map(label => dataPoints[label]),
                backgroundColor: 'rgba(53, 162, 235, 0.5)',
                borderColor: 'rgba(53, 162, 235, 1)',
                barThickness: 20
            },
        ],
    };
};

const options = {
    maintainAspectRatio: true,
    responsive: true,
    scales: {
        y: {
            type: 'logarithmic' as const,
            // A logarithmic axis cannot include zero (log(0) is undefined) and
            // beginAtZero is not supported for it, so the axis starts at 1.
            min: 1,
        },
    },
};

const BriefSection: React.FC<{ data: Results, keyName: string }> = ({data, keyName}) => {
    const specificData = data[keyName];
    if (!specificData || typeof specificData.outliers === 'undefined') {
        return <div>Error: Data for the specified key name does not exist.</div>;
    }

    const chartData = processData(specificData.outliers);

    return (
        <div className="w-full md:w-1/2 lg:w-1/3 xl:w-1/2 h-96">
            <Bar data={chartData} options={options}/>
        </div>
    );
};

export default BriefSection;
