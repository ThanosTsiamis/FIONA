import React, {createContext, useCallback, useEffect, useState} from 'react';

const STORAGE_KEY = 'fiona-filename';

interface UploadContextProps {
  filename: string | null;
  setFilename: (filename: string | null) => void;
  /** False until sessionStorage has been read, so /results does not treat a refresh as "no file". */
  hydrated: boolean;
}

const UploadContext = createContext<UploadContextProps>({
  filename: null,
  setFilename: () => {},
  hydrated: false,
});

function UploadProvider({ children }: { children: React.ReactNode }) {
  const [filename, setFilenameState] = useState<string | null>(null);
  const [hydrated, setHydrated] = useState(false);

  useEffect(() => {
    setFilenameState(sessionStorage.getItem(STORAGE_KEY));
    setHydrated(true);
  }, []);

  const setFilename = useCallback((name: string | null) => {
    setFilenameState(name);
    if (name) {
      sessionStorage.setItem(STORAGE_KEY, name);
    } else {
      sessionStorage.removeItem(STORAGE_KEY);
    }
  }, []);

  const value = {
    filename,
    setFilename,
    hydrated,
  };

  return (
    <UploadContext.Provider value={value}>{children}</UploadContext.Provider>
  );
}

export { UploadContext, UploadProvider };
