import React from 'react';
import Link from 'next/link';

interface PageButtonProps {
    href: string;
    label: string;
    icon: string;
    iconLabel: string;
}

const PageButton: React.FC<PageButtonProps> = ({ href, label, icon, iconLabel }) => (
    <div className="border border-gray-200 rounded-md p-4 max-w-xs absolute top-8 right-8">
        <p className="text-lg font-semibold">
            <Link href={href}>
                <a className="text-gray-800 no-underline hover:underline">{label}</a>
            </Link>
            <span role="img" aria-label={iconLabel}>{icon}</span>
        </p>
    </div>
);

export default PageButton;
