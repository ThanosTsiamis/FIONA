import Image from "next/image";
import Link from "next/link";

export default function Ouch500() {
    return <>
        <h1 className={"font-extrabold text-gray-900 lg:text-7xl dark:text-white"}>OOPS! 500</h1>
        <div className="flex items-center">
            <Image src="/resources/meltedEarth.png" alt="Melted Earth" width={500} height={500}/>

            <span>
                Our servers encountered a glitch.
                Rerun the algorithm and if it persists please ask for help.
            </span>
        </div>
        <Link href="/">
            <a>
                Why don't you check the <u>homepage</u>?
            </a>
        </Link>
    </>
}