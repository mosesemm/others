package za.co.mashandiro.springplay.controllers;

import lombok.extern.slf4j.Slf4j;

@Slf4j
public class FirstThings {

    public static void main(String[] args) {
        log.info("Hello World!");

        log.warn("this is a warnning: {} - {}", "banna", "wakwa akere");

        log.atInfo().log("this is a log");

        log.atDebug().setMessage("Some debug message {} - {} ")
            .addArgument(1).addArgument(2).log();
        
        log.info("debug enabled...{}", log.isDebugEnabled());
    }

}
