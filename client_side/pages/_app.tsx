import React from 'react';
import type {AppProps} from 'next/app';
import {UploadProvider} from '../components/UploadContext';

import '../styles/globals.css';

function MyApp({Component, pageProps}: AppProps) {
    return (
        <UploadProvider>
            <Component {...pageProps} />
        </UploadProvider>
    );
}

export default MyApp;
