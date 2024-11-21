#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 09:15:38 2024
"""

import pandas as pd
import numpy as np
import math

def loan_cashflow(df):
    """
    

    Parameters
    ----------
    dd (pd.DataFrame): DataFrame containing loan details with columns:
            - loan_age (int): Age of the loan in months.
            - gross_loan_rate (float): Annual interest rate of the loan.
            - loan_term_origination (int): Original term of the loan in months.
            - loan_grade (str): Grade of the loan.
            - original_balance (float): Original loan amount.
            - current_balance (float): Remaining loan balance.

    Returns:(pd.DataFrame)
           - principal
           - interest
           - payment
        
    -------

    """
    
    def monthly_payment(current_balance, gross_loan_rate, loan_term_origination):
        """
        Helper function to calculate monthly payment of each loan

        Parameters
        ----------
        
            current_balance (float): Original loan amount
            gross_loan_rate (float): Annual interest rate of the loan
            loan_term_origination (int): Original term of the loan in months

        Returns:
            int: A number for each loan
        -------
        

        """
        monthly_rate = gross_loan_rate / 12
        return current_balance * (monthly_rate * (1 + monthly_rate) ** loan_term_origination) / ((1 + monthly_rate) ** loan_term_origination - 1)
    
    def check_missing_non_numeric(row):
        """
        Helper function to check missing values or non-numeric values

        Parameters
        ----------
        row : df.DataFrame

        Returns:
            df.DataFrame that contains problematic numbers
        """
        # check missing values
        has_missing = row.isna().any()
        # check non-numeric values
        has_non_numeric = row.apply(lambda x: pd.to_numeric(x, errors='coerce')).isna().any()
        return has_missing or has_non_numeric
    
    # Calculate loan cashflows
    loan_df = []
    numeric_cols = ['loan_age', 'gross_loan_rate', 'loan_term_origination', 'current_balance']
    
    # Check if input data has values
    if df.empty:
        raise ValueError("Input DataFrame is empty. Please provide valid data.")
    else:
        print("Calculating loan cashflow")
    # Store the rows that have missing info for later use
    miss_df = df[numeric_cols].apply(check_missing_non_numeric, axis=1)
    
    # remove the rows from original data that has misisng info
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df = df.dropna()
    
    for _, row in df.iterrows():
        # if any row has missing number or non-numeric values, log the row
        remaining_term = row['loan_term_origination'] - row['loan_age']
        if row['current_balance'] <= 0:
            # when loan is fully paid off
            cashflow = {"principal": 0, "interest": 0, "total": 0}
        elif remaining_term <= 0:
            # when loan's age is greater than loan's term, and the remaining balance is not zero. 
            # Either set the loan delinquent or extended. the below scenario is delinquent
            cashflow = {"principal": 0, "interest": 0, "total": 0}           
        else:
            pmt = monthly_payment(row['current_balance'], row['gross_loan_rate'], remaining_term)
            interest = row['current_balance'] * row['gross_loan_rate'] / 12
            
            # if monthly payment is larger than the current balance after paying interest, only pay up the current balance
            principal = min(pmt - interest, row['current_balance'])
            cashflow = {"principal": principal, "interest": interest, "total": pmt}
        loan_df.append(cashflow)
        
    return pd.DataFrame(loan_df)


def collateral_cashflow(df):
    """

    Parameters
    ----------
    df : pd.DataFrame has cashflow of each loan

    Returns: (float)
             total cashflow
    -------

    """
    # Check if input has data
    if df.empty:
        raise ValueError("Input DataFrame is empty. Please provide valid data.")
    else:
        print("Calculating collateral cashflow")
        
    collateral_dict = {
        "total_principal": df['principal'].sum(),
        "total_interest": df['interest'].sum(),
        "total_cashflow": df['total'].sum()
        }
    collateral_cf = collateral_dict['total_cashflow']
    return collateral_cf

def waterfall_dist(collateral_cf, fee, tranche_weights):
    if math.isnan(collateral_cf) or collateral_cf < 0:
        raise ValueError("Input cashflow is not valid. Please provide valid data.")
    else:
        print("Calculating loan waterfall distribution")
        
    """
    Calculate the waterfall distribution of a given collateral cashflow.
    
    Parameters:
        collateral_cashflow (float): Total cashflow available for distribution.
        fee (float): Fee deducted before distributing to tranches.
        tranche_weights (dict): Weights for each tranche as percentages (e.g., {"senior": 0.7, "mezzanine": 0.2, "equity": 0.1}).
    
    Returns:
        dict: A dictionary with allocated cashflows for each tranche and the remaining cashflow.
    """
    # Deduct fees from the total cashflow
    available_cashflow = collateral_cf - fee
    if available_cashflow < 0:
        raise ValueError("Collateral cashflow is insufficient to cover the fees.")
    
    # Initialize waterfall distribution
    distribution = {}
    remaining_cashflow = available_cashflow
    
    # Allocate cashflow to each tranche based on weights
    for tranche, weight in tranche_weights.items():
        allocated_cashflow = min(remaining_cashflow, available_cashflow * weight)
        distribution[tranche] = allocated_cashflow
        remaining_cashflow -= allocated_cashflow
    
    # Add remaining cashflow to the result
    distribution["remaining_cashflow"] = remaining_cashflow
    
    return distribution



# Example usage
data = {
    "loan_age": [12, 24, 6],
    "gross_loan_rate": [4.5, 3.75, 5.0],
    "loan_term_origination": [360, 240, 120],
    "loan_grade": ["A", "B", "C"],
    "original_balance": [300000, 200000, 150000],
    "current_balance": [290000, 190000, 145000]
}
fee = 500
# Distribution weights
tranche_weights = {"senior": 0.7, 
                   "mezzanine": 0.2, 
                   "equity": 0.1
                   }  


df = pd.DataFrame(data)
loan_df = loan_cashflow(df)
collateral_cf = collateral_cashflow(loan_df)
dist = waterfall_dist(collateral_cf, fee, tranche_weights)
# Print results
print("Loan Cashflows:", loan_df)
print("Collateral Cashflows:", collateral_cf)
print("Waterfall Distribution:", dist)
