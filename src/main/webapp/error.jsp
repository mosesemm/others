<%@ taglib prefix="s" uri="/struts-tags" %>
<%@ page language="java" contentType="text/html; charset=UTF-8" pageEncoding="UTF-8" %>
<!DOCTYPE html>
<html>
    <head>
        <meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
        <s:head />
    </head>
    <body>
        <h4>Unexpected error occurred.</h4>
        <p>Please contact technical support with the following information:</p>
        <h4>Exception Name: <s:property value="exception"/></h4>
        <h4>Exception details: <s:property value="exceptionStack"/></h4>
    </body>
</html>