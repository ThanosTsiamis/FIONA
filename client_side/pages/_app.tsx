import React from 'react';
import type {AppProps} from 'next/app';
import Head from 'next/head';
import {UploadProvider} from '../components/UploadContext';

import '../styles/globals.css';

function MyApp({Component, pageProps}: AppProps) {
    return (
        <>
            <Head>
                <title>FIONA: Categorical Outlier Detector</title>
                <meta name="description"
                      content="FIONA finds categorical outliers in datasets by analysing the syntactic structure of each attribute."/>
                <meta name="theme-color" content="#ffffff"/>
                <link rel="manifest" href="/site.webmanifest"/>
                <link rel="msapplication-config" href="/browserconfig.xml"/>
            </Head>
            <UploadProvider>
                <Component {...pageProps} />
            </UploadProvider>
        </>
    );
}

export default MyApp;
