import { useState, useEffect } from "react";

export default function TableWithSlider() {
  const [visibleRows, setVisibleRows] = useState(3);
  const [tableData, setTableData] = useState([]);

  useEffect(() => {
    const fetchTableData = async () => {
      const response = await fetch("/api/fetch");
      const data = await response.json();
      setTableData(data);
    };

    fetchTableData();
  }, []);

  const handleSliderChange = (event) => {
    setVisibleRows(Number(event.target.value));
  };

  const renderTableRows = () => {
    return tableData.slice(0, visibleRows).map((data) => (
      <tr key={data.id}>
        <td>{data.name}</td>
        <td>{data.age}</td>
      </tr>
    ));
  };

  return (
    <div>
      <h1>Table with Slider</h1>
      <input
        type="range"
        min="1"
        max={tableData.length}
        value={visibleRows}
        onChange={handleSliderChange}
      />
      <span>{visibleRows}</span>
      <table>
        <thead>
          <tr>
            <th>Name</th>
            <th>Age</th>
          </tr>
        </thead>
        <tbody>{renderTableRows()}</tbody>
      </table>
    </div>
  );
}
