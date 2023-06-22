import Link from 'next/link'
import Image from 'next/image'

export default function FourOFour() {
    return <>
        <h1 className={"font-extrabold text-gray-900 lg:text-7xl dark:text-white"}>404</h1>
        <div className="flex items-center">
            <Image src="/resources/404.png" alt="A folder with a frown face" width={500} height={500}/>

            <span>Oops! We're sorry, but the page you're looking for is nowhere to be found. <br/>
                It's either been abducted by aliens,
                eaten by the Kraken, or simply vanished into thin air.<br/>
                Rest assured, our team of highly-trained web ninjas has
                been dispatched to search high and low for the missing page.</span>
        </div>
        <Link href="/">
            <a>
                Why don't you check the <u>homepage</u>?
            </a>
        </Link>
    </>
}
