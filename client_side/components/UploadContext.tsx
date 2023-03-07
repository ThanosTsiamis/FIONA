import React, { createContext, useState } from 'react';

interface UploadContextProps {
  filename: string | null;
  setFilename: (filename: string | null) => void;
}

const UploadContext = createContext<UploadContextProps>({
  filename: null,
  setFilename: () => {},
});

function UploadProvider({ children }: { children: React.ReactNode }) {
  const [filename, setFilename] = useState<string | null>(null);

  const value = {
    filename,
    setFilename,
  };

  return (
    <UploadContext.Provider value={value}>{children}</UploadContext.Provider>
  );
}

export { UploadContext, UploadProvider };
