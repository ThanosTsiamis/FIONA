import Image from "next/image";
import Link from "next/link";

// Pages Router custom error page. A file named `pages/error.tsx` would only create
// an `/error` route; Next only treats `pages/_error.tsx` as the error boundary.
export default function ErrorPage() {
    return <>
        <h1 className={"font-extrabold text-gray-900 lg:text-7xl dark:text-white"}>OOPS!</h1>
        <div className="flex items-center">
            <Image src="/resources/meltedEarth.png" alt="Melted Earth" width={500} height={500}/>

            <span>
                Our servers encountered a glitch.
                Rerun the algorithm and if it persists please ask for help.
            </span>
        </div>
        <Link href="/">
            Why don&apos;t you check the <u>homepage</u>?
        </Link>
    </>
}
