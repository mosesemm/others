Feature: Cheese finder
    This is to help users find Cheese

    Scenario: Finding some cheese
        Given I am on the google search page
        When I search for "Cheese!"
        Then the page title should start with "cheese"