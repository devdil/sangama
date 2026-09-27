on run
    set appPath to POSIX path of (path to me)
    set worker to appPath & "Contents/Resources/sangama"
    display dialog "Start an invited Sangama worker using the node configuration supplied by your operator. Your private keys stay on this computer. Prepared model shards and redeemed membership are required." buttons {"Cancel", "Choose configuration"} default button "Choose configuration"
    set configFile to choose file with prompt "Choose your Sangama worker JSON configuration"
    set commandText to quoted form of worker & " mesh --config " & quoted form of POSIX path of configFile
    tell application "Terminal"
        activate
        do script commandText
    end tell
end run
