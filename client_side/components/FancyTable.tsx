import React from 'react';

interface TableProps {
  csvData: string[][];
}

// Rendering an entire 1 MB CSV as DOM rows can freeze the page. The preview is
// only meant to confirm that the file was parsed correctly, so cap it.
const MAX_PREVIEW_ROWS = 100;

const FancyTable: React.FC<TableProps> = ({ csvData }) => {
  const rows = csvData.slice(1, MAX_PREVIEW_ROWS + 1);
  const totalRows = Math.max(csvData.length - 1, 0);

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full divide-y divide-gray-200">
        <thead className="bg-gray-50">
          <tr>
            {csvData[0]?.map((header, index) => (
              <th
                key={index}
                className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
              >
                {header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="bg-white divide-y divide-gray-200">
          {rows.map((row, rowIndex) => (
            <tr key={rowIndex} className="hover:bg-gray-100">
              {row.map((cell, cellIndex) => (
                <td
                  key={cellIndex}
                  className="px-6 py-4 whitespace-normal text-sm text-gray-900"
                >
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {totalRows > MAX_PREVIEW_ROWS && (
        <p className="px-6 py-3 text-sm text-gray-500">
          Showing the first {MAX_PREVIEW_ROWS} of {totalRows} rows.
        </p>
      )}
    </div>
  );
};

export default FancyTable;
